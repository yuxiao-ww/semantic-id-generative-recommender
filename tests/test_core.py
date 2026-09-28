import json
import tempfile
import unittest
from pathlib import Path

import torch

from gr.dataset import sid_to_tokens, tokens_to_sid
from gr.trie import SidTrie
from semantic_id.rqvae import ResidualVQ


class CoreInvariantTests(unittest.TestCase):
    def test_semantic_id_token_round_trip(self):
        sid = [3, 17, 201, 4]
        self.assertEqual(tokens_to_sid(sid_to_tokens(sid)), sid)

    def test_trie_allows_only_catalog_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "item_sid.json"
            path.write_text(
                json.dumps({"1": [3, 7, 9, 0], "2": [3, 8, 4, 0]})
            )
            trie = SidTrie(path)

            first = sid_to_tokens([3, 7, 9, 0])
            second = sid_to_tokens([3, 8, 4, 0])
            self.assertEqual(trie.get_valid_next([]), {first[0]})
            self.assertEqual(trie.get_valid_next(first[:1]), {first[1], second[1]})
            self.assertEqual(trie.get_item(first), 1)
            self.assertIsNone(trie.get_item([999, 999, 999, 999]))

    def test_residual_quantizer_shape_and_gradients(self):
        quantizer = ResidualVQ(n_layers=3, num_codes=8, code_dim=4)
        latent = torch.randn(5, 4, requires_grad=True)
        quantized, indices, loss = quantizer(latent)

        self.assertEqual(quantized.shape, latent.shape)
        self.assertEqual(indices.shape, (5, 3))
        self.assertTrue(bool(torch.all((indices >= 0) & (indices < 8))))
        loss.backward()
        self.assertIsNotNone(latent.grad)


if __name__ == "__main__":
    unittest.main()
