#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

void thread_entry() {
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(1, &mask);

	if(pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
		perror("pthread_setaffinity_np");
	}


}

int main() {
	long numCpuAvailable = sysconf(_SC_NPROCESSORS_CONF);
    printf("system cpu num is %ld\n", numCpuAvailable);
    printf("system enable cpu num is %ld\n", sysconf(_SC_NPROCESSORS_ONLN));


    for(int i = 0; i < 10; i++)
    pthread_t pid;

    if(pthread_create(&pid, NULL, thread_entry, NULL) != 0) {
    	perror("pthread_create");
    }


    return 0;
}