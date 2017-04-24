/*
* Copyright (c) Newcastle University, UK.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* 1. Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

// Threads
// Dan Jackson, 2014

#ifndef THREAD_H
#define THREAD_H

// Threads
typedef unsigned int thread_attr_t;
typedef unsigned int thread_return_t;
#define thread_return_value(value) ((unsigned int)(value))
#ifdef _WIN32
	#define THREAD_CALL __stdcall
	typedef unsigned int(__stdcall *thread_start_routine_t)(void *);
	typedef void *thread_t;
	typedef void *mutex_t;
	typedef void *event_t;
#else
	#define THREAD_CALL
	typedef unsigned int (*thread_start_routine_t)(void *);
	#include <pthread.h>

	typedef pthread_t thread_t;
	typedef pthread_mutex_t mutex_t;
	typedef struct
	{
		volatile bool signalled;
		pthread_mutex_t mutex;
		pthread_cond_t cond;
	} pevent_t;
	typedef pevent_t *event_t;
#endif

// Thread methods
int thread_create(thread_t *thread, const thread_attr_t *attr, thread_start_routine_t start_routine, void *arg);
int thread_join(thread_t thread, void **retval);
int thread_cancel(thread_t thread);

// Mutex
int mutex_init(mutex_t *mutex, int type);
int mutex_lock(mutex_t *mutex);
int mutex_timedlock_relative_msec(mutex_t *mutex, int timeout);
int mutex_unlock(mutex_t *mutex);
int mutex_destroy(mutex_t *mutex);

// Event
int event_init(event_t *event);
int event_signal(event_t *event);
int event_wait(event_t *event);
int event_destroy(event_t *event);


#endif
