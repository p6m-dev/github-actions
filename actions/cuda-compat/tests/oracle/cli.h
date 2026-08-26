/* Minimal stand-in for libnvidia-container's cli.h: only what dsl.c/dsl.h need. */
#ifndef STUB_CLI_H
#define STUB_CLI_H
#include <stdio.h>
#include <string.h>
struct error { char *msg; };
struct nvc_driver_info { char *nvrm_version; char *cuda_version; };
struct nvc_device { char *arch; char *brand; };
#define error_setx(err, fmt, ...) do { char b[512]; snprintf(b, sizeof b, fmt, ##__VA_ARGS__); (err)->msg = strdup(b); } while (0)
#endif
