// COPIED FROM IPTSD (https://github.com/linux-surface/iptsd)

/* SPDX-License-Identifier: GPL-2.0-or-later */

#ifndef _IPTSD_UTILS_H_
#define _IPTSD_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define iptsd_err(ERRNO, ...) iptsd_utils_err(ERRNO, __FILE__, __LINE__, __VA_ARGS__)

int iptsd_utils_open(const char *file, int flags);
int iptsd_utils_close(int fd);
int iptsd_utils_read(int fd, void *buf, size_t count);
int iptsd_utils_write(int fd, void *buf, size_t count);
int iptsd_utils_ioctl(int fd, unsigned long request, void *data);
int iptsd_utils_signal(int signum, void (*handler)(int));
void iptsd_utils_err(int err, const char *file, int line, const char *format, ...);
int iptsd_utils_msec_timestamp(uint64_t *ts);
int iptsd_utils_msleep(uint64_t msecs);

#ifdef __cplusplus
}
#endif

#endif /* _IPTSD_UTILS_H_ */
