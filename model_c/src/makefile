SCRIPTS_DIR=../scripts
BIN_DIR=../bin

CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops -Wno-unused-result

all: hin2vec

hin2vec: hin2vec.c
	$(CC) hin2vec.c -o ${BIN_DIR}/hin2vec $(CFLAGS)

clean:
	pushd ${BIN_DIR} && rm -rf hin2vec; popd
