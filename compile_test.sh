
#PATH_TO_BOOST_LIB=/usr/lib/x86_64-linux-gnu/
PATH_TO_BOOST_LIB=/usr/local/lib/
PATH_TO_BOOST_INCLUDE=/usr/include/boost/

set -x
nvcc -arch=sm_37 -std=c++11 -I $PATH_TO_BOOST_INCLUDE ${PATH_TO_BOOST_LIB}libboost_program_options.a -O3 -lcublas -lcurand ./src/main.cu -o LongTest
