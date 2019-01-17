#!/usr/bin/env python3

import random
import time
import unittest

import numpy as np
import tensorflow as tf

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from utils import rewards_to_discounted_returns, Timer
from utils_tensorflow import make_copy_ops, set_random_seeds, logit_entropy


class TestMiscUtils(unittest.TestCase):

    def test_returns_easy(self):
        r = [0, 0, 0, 5]
        discounted_r = rewards_to_discounted_returns(r, discount_factor=0.99)
        np.testing.assert_allclose(discounted_r,
                                   [0.99 ** 3 * 5,
                                    0.99 ** 2 * 5,
                                    0.99 ** 1 * 5,
                                    0.99 ** 0 * 5])

    def test_returns_hard(self):
        r = [1, 2, 3, 4]
        discounted_r = rewards_to_discounted_returns(r, discount_factor=0.99)
        expected = [1 + 0.99 * 2 + 0.99 ** 2 * 3 + 0.99 ** 3 * 4,
                    2 + 0.99 * 3 + 0.99 ** 2 * 4,
                    3 + 0.99 * 4,
                    4]
        np.testing.assert_allclose(discounted_r, expected)

    def test_timer(self):
        timer = Timer(duration_seconds=1)

        timer.reset()
        self.assertEqual(timer.done(), False)
        time.sleep(0.8)
        self.assertEqual(timer.done(), False)
        time.sleep(0.3)
        self.assertEqual(timer.done(), True)
        time.sleep(0.3)
        self.assertEqual(timer.done(), True)

        timer.reset()

        self.assertEqual(timer.done(), False)
        time.sleep(0.9)
        self.assertEqual(timer.done(), False)
        time.sleep(0.2)
        self.assertEqual(timer.done(), True)

    def test_random_seed(self):
        # Note: TensorFlow random seeding doesn't work completely as expected.
        # tf.set_random_seed sets a the graph-level seed in the current graph.
        # But operations also have their own operation-level seed, which is chosen
        # deterministically based on the graph-level seed, but also based on other things.
        #
        # So if you create multiple operations in the same graph, each one will be given a
        # different operation-level seed. The graph-level seed just determines what the sequence of
        # operation-level seeds will be.
        #
        # To get a bunch of operations with the same sequence of operation-level seeds, we need to
        # reset the graph before creation of each bunch of operations.

        # Generate some random numbers from a specific seed
        tf.reset_default_graph()
        sess = tf.Session()
        set_random_seeds(0)
        tf_rand_var = tf.random_normal([10])
        numpy_rand_1 = np.random.rand(10)
        numpy_rand_2 = np.random.rand(10)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 numpy_rand_1, numpy_rand_2)
        tensorflow_rand_1 = sess.run(tf_rand_var)
        tensorflow_rand_2 = sess.run(tf_rand_var)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 tensorflow_rand_1, tensorflow_rand_2)
        python_rand_1 = [random.random() for _ in range(10)]
        python_rand_2 = [random.random() for _ in range(10)]
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 python_rand_1, python_rand_2)

        # Put the seed back and check we get the same numbers
        tf.reset_default_graph()
        sess = tf.Session()
        set_random_seeds(0)
        tf_rand_var = tf.random_normal([10])
        numpy_rand_3 = np.random.rand(10)
        numpy_rand_4 = np.random.rand(10)
        np.testing.assert_equal(numpy_rand_1, numpy_rand_3)
        np.testing.assert_equal(numpy_rand_2, numpy_rand_4)
        tensorflow_rand_3 = sess.run(tf_rand_var)
        tensorflow_rand_4 = sess.run(tf_rand_var)
        np.testing.assert_equal(tensorflow_rand_1, tensorflow_rand_3)
        np.testing.assert_equal(tensorflow_rand_2, tensorflow_rand_4)
        python_rand_3 = [random.random() for _ in range(10)]
        python_rand_4 = [random.random() for _ in range(10)]
        np.testing.assert_equal(python_rand_1, python_rand_3)
        np.testing.assert_equal(python_rand_2, python_rand_4)

        # Set a different seed and make sure we get different numbers
        set_random_seeds(1)
        numpy_rand_5 = np.random.rand(10)
        numpy_rand_6 = np.random.rand(10)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 numpy_rand_5, numpy_rand_1)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 numpy_rand_6, numpy_rand_2)
        tensorflow_rand_5 = sess.run(tf_rand_var)
        tensorflow_rand_6 = sess.run(tf_rand_var)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 tensorflow_rand_5, tensorflow_rand_1)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 tensorflow_rand_6, tensorflow_rand_2)
        python_rand_5 = [random.random() for _ in range(10)]
        python_rand_6 = [random.random() for _ in range(10)]
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 python_rand_5, python_rand_1)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 python_rand_6, python_rand_2)


class TestEntropy(unittest.TestCase):

    def setUp(self):
        self.sess = tf.Session()

    def test_basic(self):
        """
        Manually calculate entropy, and check the result matches.
        """
        logits = np.array([1., 2., 3., 4.])
        probs = np.exp(logits) / np.sum(np.exp(logits))
        expected_entropy = -np.sum(probs * np.log(probs))
        actual_entropy = self.sess.run(logit_entropy(logits))[0]
        np.testing.assert_approx_equal(actual_entropy, expected_entropy, significant=5)

    def test_stability(self):
        """
        Test an example which would normally break numerical stability.
        """
        logits = np.array([0., 1000.])
        expected_entropy = 0.
        actual_entropy = self.sess.run(logit_entropy(logits))
        np.testing.assert_approx_equal(actual_entropy, expected_entropy, significant=5)

    def test_batch(self):
        """
        Make sure we get the right result if calculating entropies on a batch of probabilities.
        """
        # shape is (2, 4) (where the first dimension is the batch size)
        logits = np.array([[1., 2., 3., 4.],
                           [1., 2., 2., 1.]])
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        expected_entropy = -np.sum(probs * np.log(probs), axis=1, keepdims=True)
        actual_entropy = self.sess.run(logit_entropy(logits))
        np.testing.assert_allclose(actual_entropy, expected_entropy, atol=1e-4)

    def test_gradient_descent(self):
        """
        Check that if we start with a distribution and use gradient descent to maximise entropy,
        we end up with a maximum-entropy distribution.
        """
        logits = tf.Variable([1., 2., 3., 4., 5.])
        neg_ent = -logit_entropy(logits)
        train_op = tf.train.AdamOptimizer().minimize(neg_ent)
        self.sess.run(tf.global_variables_initializer())
        for i in range(10000):
            self.sess.run(train_op)
        expected = [0.2, 0.2, 0.2, 0.2, 0.2]  # maximum-entropy distribution
        actual = self.sess.run(tf.nn.softmax(logits))
        np.testing.assert_allclose(actual, expected, atol=1e-4)


class TestCopyNetwork(unittest.TestCase):

    def test(self):
        sess = tf.Session()

        inits = {}
        inits['from_scope'] = {}
        inits['to_scope'] = {}
        inits['from_scope']['w1'] = np.array([1.0, 2.0]).astype(np.float32)
        inits['from_scope']['w2'] = np.array([3.0, 4.0]).astype(np.float32)
        inits['to_scope']['w1'] = np.array([5.0, 6.0]).astype(np.float32)
        inits['to_scope']['w2'] = np.array([7.0, 8.0]).astype(np.float32)

        scopes = ['from_scope', 'to_scope']

        variables = {}
        for scope in scopes:
            with tf.variable_scope(scope):
                w1 = tf.Variable(inits[scope]['w1'], name='w1')
                w2 = tf.Variable(inits[scope]['w2'], name='w2')
                variables[scope] = {'w1': w1, 'w2': w2}
        copy_ops = make_copy_ops(from_scope='from_scope', to_scope='to_scope')

        sess.run(tf.global_variables_initializer())

        # Check that the variables start off being what we expect them to.
        for scope in scopes:
            for var_name, var in variables[scope].items():
                actual = sess.run(var)
                if 'w1' in var_name:
                    expected = inits[scope]['w1']
                elif 'w2' in var_name:
                    expected = inits[scope]['w2']
                np.testing.assert_equal(actual, expected)

        sess.run(copy_ops)

        # Check that the variables in from_scope are untouched.
        for var_name, var in variables['from_scope'].items():
            actual = sess.run(var)
            if 'w1' in var_name:
                expected = inits['from_scope']['w1']
            elif 'w2' in var_name:
                expected = inits['from_scope']['w2']
            np.testing.assert_equal(actual, expected)

        # Check that the variables in to_scope have been modified.
        for var_name, var in variables['to_scope'].items():
            actual = sess.run(var)
            if 'w1' in var_name:
                expected = inits['from_scope']['w1']
            elif 'w2' in var_name:
                expected = inits['from_scope']['w2']
            np.testing.assert_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
