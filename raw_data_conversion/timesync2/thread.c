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

// Cross-platform multi-threading and mutex
#ifdef _WIN32

#define _CRT_SECURE_NO_DEPRECATE
#define _WIN32_DCOM
#include <windows.h>

#include "thread.h"


// Thread
int thread_create(thread_t *thread, const thread_attr_t *attr, thread_start_routine_t start_routine, void *arg)
{
	return ((*(thread) = (void *)CreateThread(0, 0, start_routine, arg, 0, NULL)) == NULL);
}

int thread_join(thread_t thread, void **retval)
{
	return (WaitForSingleObject((HANDLE)thread, INFINITE) != WAIT_OBJECT_0);
}

int thread_cancel(thread_t thread)
{
	return (TerminateThread((HANDLE)thread, -1) == 0);
}


// Mutex
int mutex_init(mutex_t *mutex, int type)
{
	return ((*mutex = (void *)CreateMutex(NULL, FALSE, NULL)) == NULL);
}

int mutex_lock(mutex_t *mutex)
{
	return (WaitForSingleObject((HANDLE)*mutex, INFINITE) != WAIT_OBJECT_0);
}

int mutex_timedlock_relative_msec(mutex_t *mutex, int timeout)
{
	return (WaitForSingleObject((HANDLE)*mutex, timeout) != WAIT_OBJECT_0);
}

int mutex_unlock(mutex_t *mutex)
{
	return (ReleaseMutex((HANDLE)*mutex) == 0);
}

int mutex_destroy(mutex_t *mutex)
{
	return (CloseHandle((HANDLE)*mutex) == 0);
}


// Event
int event_init(event_t *event)
{
	return ((*event = (void *)CreateEvent(NULL, FALSE, FALSE, NULL)) == NULL);
}

int event_signal(event_t *event)
{
	return (SetEvent((HANDLE)*event) == 0);
}

int event_wait(event_t *event)
{
	return (WaitForSingleObject((HANDLE)*event, INFINITE) != WAIT_OBJECT_0);
}

int event_destroy(event_t *event)
{
	return (CloseHandle((HANDLE)*event) == 0);
}


#else

// Headers
#include <stdlib.h>
#include <string.h>
//#include <unistd.h>
//#include <sys/wait.h>
//#include <sys/types.h>
//#include <termios.h>
#include <pthread.h>
//#include <libudev.h>
#include <stdbool.h>

#include "thread.h"


// Thread
int thread_create(thread_t *thread, const thread_attr_t *attr, thread_start_routine_t start_routine, void *arg)
{
	typedef void *(*start_routine_t)(void *);
	return pthread_create((pthread_t *)thread, (pthread_attr_t *)attr, (start_routine_t)start_routine, arg);
}

int thread_join(thread_t thread, void **retval)
{
	return pthread_join((pthread_t)thread, retval);
}

int thread_cancel(thread_t thread)
{
	return pthread_cancel((pthread_t)thread);
}


// Mutex
int mutex_init(mutex_t *mutex, int type)
{
	return pthread_mutex_init((pthread_mutex_t *)mutex, (const pthread_mutexattr_t *)&type);
}

int mutex_lock(mutex_t *mutex)
{
	return pthread_mutex_lock((pthread_mutex_t *)mutex);
}

int mutex_unlock(mutex_t *mutex)
{
	return pthread_mutex_unlock((pthread_mutex_t *)mutex);
}

int mutex_destroy(mutex_t *mutex)
{
	return pthread_mutex_destroy((pthread_mutex_t *)mutex);
}


// Event
int event_init(event_t *event)
{
	pevent_t *pevent;
	pevent = (pevent_t *)malloc(sizeof(pevent_t));
	memset(pevent, 0, sizeof(pevent_t));

	pevent->signalled = false;
	pthread_mutex_init(&pevent->mutex, NULL);
	pthread_cond_init(&pevent->cond, NULL);

	// Output
	*(pevent_t **)event = pevent;
	return 0;
}

int event_signal(event_t *event)
{
	pevent_t *pevent = (pevent_t *)*event;
	pthread_mutex_lock(&pevent->mutex);
	pevent->signalled = true;
	pthread_mutex_unlock(&pevent->mutex);
	pthread_cond_signal(&pevent->cond);
	return pthread_cond_signal(&pevent->cond);
}

int event_wait(event_t *event)
{
	pevent_t *pevent = (pevent_t *)*event;
	pthread_mutex_lock(&pevent->mutex);
	while (!pevent->signalled)
	{
		pthread_cond_wait(&pevent->cond, &pevent->mutex);
	}
	pevent->signalled = false;
	pthread_mutex_unlock(&pevent->mutex);
	return 0;
}

int event_destroy(event_t *event)
{
	pevent_t *pevent = (pevent_t *)*event;
	pthread_mutex_destroy(&pevent->mutex);
	pthread_cond_destroy(&pevent->cond);
	return 0;
}


#endif

