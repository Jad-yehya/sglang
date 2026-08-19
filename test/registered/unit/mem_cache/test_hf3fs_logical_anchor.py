# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

import json
import os
import signal
import tempfile
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
    Hf3fsLocalMetadataClient,
)
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import HiCacheHF3FS
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class MockLogicalHostPool:
    def __init__(self, size=4, page_size=2, layout="page_first"):
        self.size = size
        self.page_size = page_size
        self.layout = layout
        self.dtype = torch.uint8
        self.kv_buffer = None

    def get_ksize_per_token(self):
        return 0


class MockSidePool:
    def __init__(
        self,
        num_pages: int = 2,
        page_size: int = 2,
        layer_num: int = 3,
        item_bytes: int = 2,
    ):
        self.page_size = page_size
        self.layer_num = layer_num
        self.item_bytes = item_bytes
        self.data = torch.arange(
            num_pages * layer_num * item_bytes, dtype=torch.uint8
        ).reshape(num_pages, layer_num, item_bytes)

    def get_ksize_per_token(self):
        return self.item_bytes

    def get_data_page(self, index, flat=True):
        page = self.data[int(index) // self.page_size]
        return page.flatten() if flat else page

    def get_dummy_flat_data_page(self):
        return torch.empty(self.layer_num * self.item_bytes, dtype=torch.uint8)

    def set_from_flat_data_page(self, index, data_page):
        self.data[int(index) // self.page_size].copy_(
            data_page.reshape(self.layer_num, self.item_bytes)
        )


class TestHF3FSLogicalAnchor(unittest.TestCase):
    def test_factory_initializes_logical_anchor_without_file_pages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "hf3fs.json")
            with open(config_path, "w") as config_file:
                json.dump(
                    {
                        "file_path_prefix": os.path.join(temp_dir, "hicache"),
                        "file_size": 64,
                        "numjobs": 1,
                        "entries": 4,
                        "metadata_server_url": "http://unused",
                    },
                    config_file,
                )
            storage_config = HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                attn_cp_rank=0,
                attn_cp_size=1,
                is_mla_model=True,
                enable_storage_metrics=False,
                is_page_first_layout=True,
                model_name="test-deepseek-v4",
                extra_config={"use_mock_hf3fs_client": True},
            )

            for layout in ("page_first", "layer_first"):
                with self.subTest(layout=layout):
                    storage_config.is_page_first_layout = layout == "page_first"
                    with (
                        patch.dict(
                            os.environ,
                            {HiCacheHF3FS.default_env_var: config_path},
                        ),
                        patch(
                            "sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs.atexit.register"
                        ),
                        patch.object(signal, "signal"),
                    ):
                        backend = StorageBackendFactory._create_builtin_backend(
                            "hf3fs",
                            HiCacheHF3FS,
                            storage_config=storage_config,
                            mem_pool_host=MockLogicalHostPool(layout=layout),
                        )

                    try:
                        self.assertTrue(backend._is_logical_anchor)
                        self.assertEqual(backend.bytes_per_page, 0)
                        self.assertEqual(backend.num_pages, 0)
                        self.assertEqual(backend.clients, [])
                    finally:
                        backend.close()

    def test_logical_anchor_uses_side_pools_for_storage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "hicache.bin")
            metadata_client = Hf3fsLocalMetadataClient()
            with (
                patch(
                    "sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs.atexit.register"
                ),
                patch.object(signal, "signal"),
            ):
                backend = HiCacheHF3FS(
                    rank=0,
                    file_path=file_path,
                    file_size=64,
                    numjobs=1,
                    bytes_per_page=0,
                    entries=4,
                    client_timeout=1,
                    dtype=torch.uint8,
                    metadata_client=metadata_client,
                    is_mla_model=True,
                    is_page_first_layout=True,
                    use_mock_client=True,
                    is_logical_anchor=True,
                )

            try:
                logical_pool = MockLogicalHostPool()
                side_pool = MockSidePool()
                transfer = PoolTransfer(
                    name=PoolName.DEEPSEEK_V4_C4,
                    host_indices=torch.tensor([0, 1, 2, 3]),
                    keys=["page-0", "page-1"],
                )

                backend.register_mem_pool_host(logical_pool)
                backend.register_mem_host_pool_v2(side_pool, transfer.name)

                self.assertEqual(backend.clients, [])
                self.assertFalse(os.path.exists(file_path))
                pool_ctx = backend._pool_storage_ctx[transfer.name]
                page_template = side_pool.get_dummy_flat_data_page()
                expected_page_bytes = (
                    page_template.numel() * page_template.element_size()
                )
                self.assertEqual(pool_ctx.bytes_per_page, expected_page_bytes)
                self.assertEqual(
                    backend.batch_exists_v2(transfer.keys, [transfer]).kv_hit_pages,
                    0,
                )

                self.assertEqual(
                    backend.batch_set_v1(transfer.keys, transfer.host_indices),
                    [True, True],
                )
                self.assertTrue(all(backend.batch_set_v2([transfer])[transfer.name]))

                hit_result = backend.batch_exists_v2(transfer.keys, [transfer])
                self.assertEqual(hit_result.kv_hit_pages, 2)
                self.assertEqual(hit_result.extra_pool_hit_pages[transfer.name], 2)

                expected = side_pool.data.clone()
                side_pool.data.zero_()
                self.assertEqual(
                    backend.batch_get_v1(transfer.keys, transfer.host_indices),
                    [True, True],
                )
                self.assertTrue(all(backend.batch_get_v2([transfer])[transfer.name]))
                self.assertTrue(torch.equal(side_pool.data, expected))

                backend.clear()
            finally:
                backend.close()


if __name__ == "__main__":
    unittest.main()
