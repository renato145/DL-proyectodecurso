import math, os, random, sys, time, logging
import numpy as np
import tensorflow as tf
import data_utils
import seq2seq_model
from six.moves import xrange

learning_rate = 0.5 # ratio de aprendizaje
learning_rate_decay_factor = 0.99 # decay del ratio de aprendizaje
max_gradient_norm = 5.0 # maxima norma de las gradientes (regularizacion)
batch_size = 64 # Tamaño de los batch de entrenamiento
size = 1024 # Tamaño de las capas del modelo
num_layers = 3 # Numero de capas
from_vocab_size = 40000 # Tamaño del vocabulario en ingles
to_vocab_size = 40000 # Tamaño del vocabulario en frances
data_dir = '/tmp' # Directorio de la data
train_dir = '/tmp' # Directorio de entrenamiento
max_train_data_size = 0 # Limite del tamaño de la data de entrenamiento (0: no limite)
steps_per_checkpoint = 200 # Cada cuantos pasos se guarda un checkpoint
use_fp16 = False # Usar precision decimal de 16 bits

# Archivos de vocabulario:
from_train_data = None
to_train_data = None
from_dev_data = None
to_dev_data = None

# Duplas de buckets a usar en el preprocesamiento de la data (ingles, frances).
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.
  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).
  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      from_vocab_size,
      to_vocab_size,
      _buckets,
      size,
      num_layers,
      max_gradient_norm,
      batch_size,
      learning_rate,
      learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(train_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model


def train():
  """Entre el modelo de traduccion de ingles a frances usando el dataset WMT."""
  if from_train_data and to_train_data:
    # Carga la data del dataset WMT.
    from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
        data_dir,
        from_train_data, to_train_data,
        from_dev_data, to_dev_data,
        from_vocab_size, to_vocab_size)
  else:
      # Prepara la data del dataset WMT.
      print("Preparing WMT data in %s" % data_dir)
      from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_wmt_data(
          data_dir, from_vocab_size, to_vocab_size)

  with tf.Session() as sess:
    # Crea el modelo.
    print("Creating %d layers of %d units." % (num_layers, size))
    model = create_model(sess, False)

    # Lee la data en buckets y computa sus tamaños.
    print ("Reading development and training data (limit: %d)."
           % max_train_data_size)
    dev_set = read_data(from_dev, to_dev)
    train_set = read_data(from_train, to_train, max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # El bucket scale es una lista de numeros incrementales de 0 a 1 que se usa
    # para seleccionar un bucket especifico. El tamaño de [scale[i], scale[i+1]]
    # es proporcional al tamaño del bucket i.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # Loop de entrenamiento.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Se escoge un bucket de acuerdo a la distribucion de la data. Se escoge
      # un numero random entre [0, 1] y se usa el correspondiente intervalo en
      # train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Obtener un batch y hacer un step de entrenamiento.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / steps_per_checkpoint
      loss += step_loss / steps_per_checkpoint
      current_step += 1

      # Cada cierto numero de pasos, se guarda un checkpoint, se muestran
      # las estadisticas y se ejecutan evaluaciones.
      if current_step % steps_per_checkpoint == 0:
        # Muestra las estadisticas del epoch previo.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Disminuye el learning rate si no hubo mejoras en el entrenamiento las
        # ultimas 3 epocas.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Se guarda un checkpoint.
        checkpoint_path = os.path.join(train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Se ejecutan las evaluaciones en el development set y se imprime la
        # perplejidad.
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
              "inf")
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))


def decode():
  with tf.Session() as sess:
    # Crea el modelo y carga los parametros.
    model = create_model(sess, True)
    model.batch_size = 1  # Se decodea una palabra a la vez.

    # Carga vocabularios en memoria.
    en_vocab_path = os.path.join(data_dir,
                                 "vocab%d.from" % from_vocab_size)
    fr_vocab_path = os.path.join(data_dir,
                                 "vocab%d.to" % to_vocab_size)
    en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    # Lee texto del usuario.
    sentence = input('> ')
    while sentence:
      # Obtiene el token id de la oracion.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
      # Asigna la oracion a un bucket adecuado.
      bucket_id = len(_buckets) - 1
      for i, bucket in enumerate(_buckets):
        if bucket[0] >= len(token_ids):
          bucket_id = i
          break
      else:
        logging.warning("Sentence truncated: %s", sentence)

      # Crea un batch con la primera palabra.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Obtiene los resultados del modelo (softmax).
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # Se asigna como output el resultado con mayor valor (argmax).
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # En caso de un token 'EOS' (end of sentence), se termina la traduccion.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Muestra el resultado final.
      print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
      sys.stdout.flush()
      sentence = input('> ')

decode = False
self_test = False

###
self_test()
###
train()

###
decode()