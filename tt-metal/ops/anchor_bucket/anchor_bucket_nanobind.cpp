// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "anchor_bucket_nanobind.hpp"
#include <nanobind/nanobind.h>
#include "ttnn-nanobind/bind_function.hpp"
#include "anchor_bucket.hpp"

namespace ttnn::operations::anchor_bucket {

void bind_anchor_bucket(nb::module_& mod) {
    ttnn::bind_function<"anchor_bucket">(
        mod,
        "Buckets anchors by camera-visibility pattern from grid_compact's "
        "flags: writes perm (sorted pos -> anchor), inv (anchor -> sorted "
        "pos) and a per-tile-row live-camera bitmap, the inputs of "
        "grouped_weighted_sum's dead-pair skip. Stable integer counting sort "
        "on one core.",
        &ttnn::anchor_bucket,
        nb::arg("flags"), nb::arg("perm"), nb::arg("inv"), nb::arg("live"),
        nb::arg("num_anchors"));
}

}  // namespace ttnn::operations::anchor_bucket
