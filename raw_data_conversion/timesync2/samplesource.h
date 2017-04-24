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

// Sample Source
// Dan Jackson

#ifndef SAMPLESOURCE_H
#define SAMPLESOURCE_H

#include <stdio.h>
#include <stdint.h>


// A source of samples
#define SAMPLE_SOURCE_CHANNELS_MAX 16
#define SAMPLE_SOURCE_MAX_META 16384
typedef struct
{
	unsigned char *buffer;
	int bufferLength;
	int bufferOffset;
	int pageSize;					// mapping/buffer page size

	unsigned int dataStartOffset;
	unsigned int numChannels;
	unsigned int numSamples;
	unsigned int sampleRate;
	float scale[SAMPLE_SOURCE_CHANNELS_MAX];
	double startTime;				// seconds since epoch

	FILE *fp;
	long fileLength;

	char infoArtist[SAMPLE_SOURCE_MAX_META];
	char infoName[SAMPLE_SOURCE_MAX_META];
	char infoComment[SAMPLE_SOURCE_MAX_META];
	char infoDate[SAMPLE_SOURCE_MAX_META];

} sample_source_t;


// Open the source of samples (.WAV file), nonzero is failure
int SampleSourceOpen(sample_source_t *sampleSource, const char *filename);

// Close the source of samples
void SampleSourceClose(sample_source_t *sampleSource);

// Returns a pointer to the specified sample index for the specified minimum number of samples, and the span/pitch (in bytes) between samples
const int16_t *SampleSourceRead(sample_source_t *sampleSource, unsigned int index, unsigned int minCount, unsigned int *count, size_t *span);

#endif
