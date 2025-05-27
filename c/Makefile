CC      := gcc
CFLAGS  := -O3 -std=c11 -I/usr/local/include -I.
LDFLAGS := -L/usr/local/lib -ltensorflowlite_c -lm -ldl -pthread

all: infer_emnist

infer_emnist: main.c emnist_train_model.h stb_image.h
	$(CC) $(CFLAGS) main.c -o $@ $(LDFLAGS)

clean:
	rm -f infer_emnist