DEVICE   ?= cuda


MODULES  := qat_zip cpu_zip iaa_zip
ifeq ($(DEVICE),cuda)
MODULES  += dsa_gd
endif


INCDIR   := include


LIBDIR   := ../../_lib
QATBUILD := ../../_build/qat
QATURL   := https://downloadmirror.intel.com/921047/QAT20.L.1.2.30-00178.tar.gz
# IAXL_QATLIB (optional): oot (default when unset/empty) = build the out-of-tree
# QAT20 package into _lib, intree = system qatlib for the in-tree kernel driver.
QATMODE  := $(or $(strip $(IAXL_QATLIB)),oot)
ifeq ($(QATMODE),intree)
ifeq ($(shell pkg-config --exists qatlib 2>/dev/null && echo 1),)
$(error IAXL_QATLIB=intree but qatlib was not found by pkg-config)
endif
QATLIBS  :=
QATCFLAGS:= -I$(shell pkg-config --variable=includedir qatlib)/qat
QATLDLIBS:= $(shell pkg-config --libs-only-L libqat libusdm) -lqat -lusdm
else ifeq ($(QATMODE),oot)
QATLIBS  := $(LIBDIR)/libqat_s.so $(LIBDIR)/libusdm_drv_s.so
QATCFLAGS:=
QATLDLIBS:= $(QATLIBS)
else
$(error IAXL_QATLIB must be oot or intree)
endif
QPLBUILD := ../../_build/qpl
QPLREPO  := https://github.com/intel/qpl.git
QPLTAG   := v1.9.0
QPLPREFIX:= $(abspath $(QPLBUILD)/install)
QPLINC   := $(QPLPREFIX)/include
QPLLIBS  := $(LIBDIR)/libqpl.so
RPATH    := -Wl,-rpath,$(abspath $(LIBDIR)) -Wl,-rpath,$(abspath .)

CC       := gcc
CXX      := g++
CFLAGS   := -O2 -g -Wall -I$(INCDIR)
LDFLAGS  := -lpthread


ARGS     ?=


qat_zip_SRCS   := qat_zip/qat_zip.c common/envs.c
qat_zip_CFLAGS := $(QATCFLAGS)
qat_zip_LDLIBS := $(QATLDLIBS)
qat_zip_DEPS   := $(QATLIBS)


cpu_zip_SRCS   := cpu_zip/cpu_zip.c common/envs.c
cpu_zip_LDLIBS := -lz


iaa_zip_SRCS   := iaa_zip/iaa_zip.c common/envs.c
iaa_zip_CFLAGS := -I$(QPLINC)
iaa_zip_LDLIBS := $(QPLLIBS) -ldl
iaa_zip_DEPS   := $(QPLLIBS)


dsa_gd_SRCS   := dsa_gd/dsa_gd.c dsa_gd/dsa_memcpy.c common/envs.c
dsa_gd_CFLAGS := -std=gnu11 -D_GNU_SOURCE -fopenmp
dsa_gd_LDLIBS := -lgdrapi -ldl -fopenmp


test_src = $(strip $(wildcard test/$(1)_test.c) $(wildcard test/$(1)_test.cpp))


LIBS         := $(foreach m,$(MODULES),$(LIBDIR)/lib$(m).so)
TESTS        := $(foreach m,$(MODULES),$(if $(call test_src,$(m)),$(m)_test))
TEST_TARGETS := $(foreach m,$(MODULES),$(if $(call test_src,$(m)),test-$(m)))
TESTS        += zip_compat_test
TEST_TARGETS += test-zip-compat
TESTS        += zip_throughput_test
TEST_TARGETS += test-zip-throughput

.PHONY: all test clean qat-userspace qpl-userspace $(TEST_TARGETS)

all: $(LIBS) $(TESTS)


$(LIBDIR)/libusdm_drv_s.so: $(LIBDIR)/libqat_s.so

$(LIBDIR)/libqat_s.so:
	mkdir -p $(QATBUILD)
	mkdir -p $(LIBDIR)
	cd $(QATBUILD) \
	    && wget -q "$(QATURL)" \
	    && tar xf *.tar.gz \
	    && export KERNEL_SOURCE_ROOT=$$(ls -dv /usr/src/linux-headers-*-generic | tail -n 1) \
	    && ./configure \
	    && sed "s/^ALL_TARGETS\ *=\ *.*/ALL_TARGETS = lac_user/g" -i quickassist/Makefile \
	    && make -j"$$(nproc)" quickassist-all
	cp -a $(QATBUILD)/build/*.so $(LIBDIR)/

$(LIBDIR)/libqpl.so:
	mkdir -p $(QPLBUILD)
	mkdir -p $(LIBDIR)
	cd $(QPLBUILD) \
	    && [ -d qpl ] || git clone -b $(QPLTAG) --depth 1 $(QPLREPO) qpl
	cd $(QPLBUILD)/qpl \
	    && cmake -B build -DCMAKE_BUILD_TYPE=Release -DQPL_LIBRARY_TYPE=SHARED \
	        -DQPL_BUILD_TESTS=OFF -DQPL_BUILD_EXAMPLES=OFF \
	        -DCMAKE_INSTALL_PREFIX="$(QPLPREFIX)" \
	    && cmake --build build -j"$$(nproc)" \
	    && cmake --install build
	cp -a $(QPLPREFIX)/lib*/libqpl.so* $(LIBDIR)/


define MODULE_template
$(1)_SRCS := $$(or $$($(1)_SRCS),$$(wildcard $(1)/*.c) $$(wildcard $(1)/*.cpp))
$(1)_CC   := $$(or $$($(1)_CC),$$(CC))
$(1)_TEST_SRC := $$(strip $$(wildcard test/$(1)_test.c) $$(wildcard test/$(1)_test.cpp))

$$(LIBDIR)/lib$(1).so: $$($(1)_SRCS) $$(wildcard $(1)/include/*.h) $$(wildcard $(1)/*.h) $$(wildcard $$(INCDIR)/*.h) $$($(1)_DEPS)
	mkdir -p $$(LIBDIR)
	$$($(1)_CC) $$(CFLAGS) $$($(1)_CFLAGS) -I$(1) -I$(1)/include -fPIC -shared -o $$@ $$($(1)_SRCS) $$($(1)_LDLIBS) $$(RPATH) $$(LDFLAGS)

ifneq ($$($(1)_TEST_SRC),)
$(1)_test: $$($(1)_TEST_SRC) $$(LIBDIR)/lib$(1).so
	$$($(1)_CC) $$(CFLAGS) $$($(1)_CFLAGS) -fopenmp -o $$@ $$< -L$$(LIBDIR) -l$(1) $$($(1)_LDLIBS) $$(RPATH) $$(LDFLAGS)

test-$(1): $(1)_test
	(cd test && numactl --cpunodebind=0 --membind=0 ../$(1)_test $$(ARGS))
endif
endef

$(foreach m,$(MODULES),$(eval $(call MODULE_template,$(m))))


zip_compat_test: test/zip_compat_test.c $(LIBDIR)/libqat_zip.so $(LIBDIR)/libcpu_zip.so $(LIBDIR)/libiaa_zip.so
	$(CC) $(CFLAGS) -o $@ $< -L$(LIBDIR) -lqat_zip -lcpu_zip -liaa_zip -lz $(RPATH) $(LDFLAGS)

test-zip-compat: zip_compat_test
	(cd test && numactl --cpunodebind=0 --membind=0 ../zip_compat_test $(ARGS))


zip_throughput_test: test/zip_throughput_test.c $(LIBDIR)/libqat_zip.so $(LIBDIR)/libiaa_zip.so
	$(CC) $(CFLAGS) $(iaa_zip_CFLAGS) -fopenmp -o $@ $< -L$(LIBDIR) -lqat_zip -liaa_zip $(RPATH) $(LDFLAGS)

test-zip-throughput: zip_throughput_test
	(cd test && numactl --cpunodebind=0 --membind=0 ../zip_throughput_test $(ARGS))


qat-userspace: $(QATLIBS)

qpl-userspace: $(QPLLIBS)


test: $(TEST_TARGETS)

clean:
	rm -f $(LIBS) $(TESTS)
