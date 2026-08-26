/* Minimal stand-in for libnvidia-container's utils.h: only what dsl.c needs. */
#ifndef STUB_UTILS_H
#define STUB_UTILS_H
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include "cli.h"
#define nitems(a) (sizeof(a) / sizeof((a)[0]))
static inline char *xstrdup(struct error *err, const char *s) { (void)err; return strdup(s); }
static inline int str_case_equal(const char *a, const char *b) { return strcasecmp(a, b) == 0; }
#endif
