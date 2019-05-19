PYTHON ?= python3

all:
	${PYTHON} setup.py build_ext -if
	rm -rf build

clean:
	rm -rf build pytrt.cpp *.so
