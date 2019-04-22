#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

#define NTHREAD	10

void *thread_entry(void *s) {
	pthread_t id = pthread_self();
	int idx = *(int *)s;

	// Bind this thread i to CPU i
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(idx, &mask);
	if(pthread_setaffinity_np(id, sizeof(mask), &mask) < 0) {
		perror("pthread_setaffinity_np");
	}

	for(int i = 0; i < 1000000; i++)
		for(int j = 0; j < 100000; j++)
			;

}

int main() {
	long numCpuAvailable = sysconf(_SC_NPROCESSORS_CONF);
    printf("system cpu num is %ld\n", numCpuAvailable);
    printf("system enable cpu num is %ld\n", sysconf(_SC_NPROCESSORS_ONLN));


    pthread_t threadId[NTHREAD];
    int threadIdx[NTHREAD];
    for(int i = 0; i < NTHREAD; i++){
    	threadIdx[i] = i;
	    if(pthread_create(&threadId[i], NULL, thread_entry, threadIdx + i) != 0) 
	    	perror("pthread_create");
    }

    for(int i = 0; i < NTHREAD; i++){
    	pthread_join(threadId[i], NULL);
    }

    return 0;
}