PROGNAME=mpi_broadcast
VERSION=1.0

MKDIR=mkdir
PACK=tar -jcvf
CC=mpicc
CFLAGS=-Wall -pedantic -std=c99 -lm
CP=cp
RM=rm -rf
PROCESSES = 40
OFILES=$(subst .c,.o,$(wildcard *.c))

$(PROGNAME): $(OFILES)
	$(CC) $(CFLAGS) $(OFILES) -o $(PROGNAME)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) *.o $(PROGNAME) $(PROGNAME)-$(VERSION).tbz

run: $(PROGNAME)
	mpirun -n $(PROCESSES) ./$(PROGNAME)

tarball: $(PROGNAME)
	$(MAKE) clean
	$(MKDIR) $(PROGNAME)-$(VERSION)
	$(CP) * $(PROGNAME)-$(VERSION) || true
	$(PACK) $(PROGNAME)-$(VERSION).tbz $(PROGNAME)-$(VERSION)/*
	$(RM) $(PROGNAME)-$(VERSION)
