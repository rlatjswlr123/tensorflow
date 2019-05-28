import tensorflow as tf

# mnist 데이터셋
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 인풋
x = tf.placeholder(tf.float32, [None, 784])

# 가중치, 바이어스
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 답을 넣기 위한 placeholder
y_ = tf.placeholder(tf.float32, [None, 10])

# cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 학습비율 0.5로 경사하강법 적용하여 크로스 엔트로피 최소화
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 변수 초기화
init = tf.global_variables_initializer()

# session에서 모델 실행
sess = tf.Session()

# 변수 초기화
sess.run(init)

# 학습 1000번
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 모델 평가하기
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 정확도 계산하기
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))