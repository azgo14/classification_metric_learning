import os
import unittest
import numpy as np

from mock import mock
from mock import call
from pyfakefs import fake_filesystem_unittest

from evaluation.retrieval import evaluate_recall_at_k
from evaluation.retrieval import evaluate_float_binary_embedding_faiss


class TestRetrieval(fake_filesystem_unittest.TestCase):

    def setUp(self):
        self.setUpPyfakefs()

        self.query_embeddings = np.array([
            [1.0, 3.0],
            [-3.0, 1.0]
        ]).astype('float32')

        self.db_embeddings = np.array([
            [2.0, 3.0],
            [1.0, 5.0],
            [-3.0, 3.0]
        ]).astype('float32')

        self.query_labels = [1, 2]

        self.db_labels = [2, 1, 2]

    def test_regular_evaluate_recall_at_k(self):
        """
            test recall_at_k
            query_db retrieval:
            q[0] -> 0, 1, 2
            q[1] -> 2, 1, 0
        """

        dists = np.array([
            [1.0, 2.0],
            [2.0, 5.39]
        ]).astype('float32')

        results = np.array([
            [0, 1],
            [2, 1]
        ])

        r_at_k = evaluate_recall_at_k(dists, results, self.query_labels, self.db_labels, 2)

        self.assertTrue(r_at_k.shape[0] == 2)
        np.testing.assert_almost_equal(r_at_k, [50.0, 100.0], 2)

    def test_self_evaluate_recall_at_k(self):
        """
            test recall_at_k
            db_db retrieval:
            db[0] -> 0, 1, 2
            db[1] -> 1, 0, 2
            db[2] -> 2, 1, 0
        """

        dists = np.array([
            [0.0, 2.24, 5.0],
            [0.0, 2.24, 4.47],
            [0.0, 4.47, 5.0]
        ]).astype('float32')

        results = np.array([
            [0, 1, 2],
            [1, 0, 2],
            [2, 1, 0]
        ])

        r_at_k = evaluate_recall_at_k(dists, results, self.db_labels, self.db_labels, 2)

        self.assertTrue(r_at_k.shape[0] == 2)
        np.testing.assert_almost_equal(r_at_k, [0.0, 66.67], 2)

    @mock.patch("evaluation.retrieval._retrieve_knn_faiss_gpu")
    @mock.patch("evaluation.retrieval.evaluate_recall_at_k")
    def test_evaluate_float_binary_embedding_faiss(self, recall_at_k_mock, nn_mock):
        output_file = "epoch_test"

        nn_mock.side_effect = [('dist1', 'result1'), ('dist2', 'result2')]

        recall_at_k_mock.side_effect = [np.zeros((1000,)), np.ones((1000,))]

        evaluate_float_binary_embedding_faiss(self.query_embeddings, self.db_embeddings,
                                              self.query_labels, self.db_labels,
                                              output_file, k=1000, gpu_id=0)

        self.assertEqual(nn_mock.call_count, 2)

        np.testing.assert_array_equal(self.query_embeddings, nn_mock.call_args_list[0][0][0])
        np.testing.assert_array_equal(self.db_embeddings, nn_mock.call_args_list[0][0][1])

        np.testing.assert_array_equal(np.require(self.query_embeddings > 0, dtype='float32'),
                                      nn_mock.call_args_list[1][0][0])
        np.testing.assert_array_equal(np.require(self.db_embeddings > 0, dtype='float32'),
                                      nn_mock.call_args_list[1][0][1])

        self.assertEqual(recall_at_k_mock.call_count, 2)
        r_at_k_calls = [
            call('dist1', 'result1', self.query_labels, self.db_labels, 1000),
            call('dist2', 'result2', self.query_labels, self.db_labels, 1000)
        ]
        recall_at_k_mock.assert_has_calls(r_at_k_calls)

        self.assertTrue(os.path.exists(output_file + '_identity.eval'))
        self.assertTrue(os.path.exists(output_file + '_binary.eval'))



if __name__ == '__main__':
    unittest.main()
