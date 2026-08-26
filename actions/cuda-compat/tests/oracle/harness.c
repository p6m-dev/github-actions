/* Differential oracle: runs libnvidia-container's own dsl_evaluate (verbatim
 * src/cli/dsl.c) with the same four rules configure.c registers, against real
 * NVIDIA_REQUIRE_CUDA strings and a stated set of node facts.
 *
 *   ./oracle <driver> <cuda> <arch> <brand> <require-expr>
 * prints SATISFIED / UNSATISFIED(reason) and exits 0 / 1.
 */
#include <stdio.h>
#include <stdlib.h>
#include "dsl.h"
#include "utils.h"

/* verbatim from libnvidia-container src/cli/configure.c */
static int check_cuda_version(const struct dsl_data *data, enum dsl_comparator cmp, const char *version)
{
        if (data->drv == NULL) return (1);
        if (data->drv->cuda_version == NULL) return (1);
        return (dsl_compare_version(data->drv->cuda_version, cmp, version));
}
static int check_driver_version(const struct dsl_data *data, enum dsl_comparator cmp, const char *version)
{
        if (data->drv == NULL) return (1);
        if (data->drv->nvrm_version == NULL) return (1);
        return (dsl_compare_version(data->drv->nvrm_version, cmp, version));
}
static int check_device_arch(const struct dsl_data *data, enum dsl_comparator cmp, const char *arch)
{
        if (data->dev == NULL) return (1);
        if (data->dev->arch == NULL) return (1);
        return (dsl_compare_version(data->dev->arch, cmp, arch));
}
static int check_device_brand(const struct dsl_data *data, enum dsl_comparator cmp, const char *brand)
{
        if (data->dev == NULL) return (1);
        if (data->dev->brand == NULL) return (1);
        return (dsl_compare_string(data->dev->brand, cmp, brand));
}
static const struct dsl_rule rules[] = {
        {"cuda", &check_cuda_version},
        {"driver", &check_driver_version},
        {"arch", &check_device_arch},
        {"brand", &check_device_brand},
};

int main(int argc, char **argv)
{
        if (argc != 6) { fprintf(stderr, "usage: %s DRIVER CUDA ARCH BRAND EXPR\n", argv[0]); return 2; }
        struct nvc_driver_info drv = { .nvrm_version = argv[1], .cuda_version = argv[2] };
        struct nvc_device dev = { .arch = argv[3], .brand = argv[4] };
        struct dsl_data data = { .drv = &drv, .dev = &dev };
        struct error err = {0};
        if (dsl_evaluate(&err, argv[5], &data, rules, nitems(rules)) < 0) {
                printf("UNSATISFIED: %s\n", err.msg ? err.msg : "(no message)");
                return 1;
        }
        printf("SATISFIED\n");
        return 0;
}
