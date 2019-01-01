set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
add_library(hash STATIC ../include/hash.h ../hash.c)
target_link_libraries(hash PRIVATE ${CMAKE_THREAD_LIBS_INIT})