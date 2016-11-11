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

// Samples
// Dan Jackson

//#define FILTER (1*200/4)
//#define NOISE 0.1f
//#define DEBUG_OUTPUT

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "samples.h"
#include "samplesource.h"
#include "exits.h"


int SamplesOpen(samples_t *samples, const char *filename)
{
	sample_source_t sampleSource;
	int ret;
	ret = SampleSourceOpen(&sampleSource, filename);
	if (ret != EXIT_OK) { return ret; }

	samples->numSamples = sampleSource.numSamples;
	samples->sampleRate = sampleSource.sampleRate;
	samples->startTime = sampleSource.startTime;

	strcpy(samples->infoArtist,  sampleSource.infoArtist);
	strcpy(samples->infoComment, sampleSource.infoComment);
	strcpy(samples->infoDate,    sampleSource.infoDate);
	strcpy(samples->infoName,    sampleSource.infoName);

	size_t length = sizeof(float) * samples->numSamples;
	float *buffer = (float *)malloc(length);
	if (buffer == NULL) { fprintf(stderr, "ERROR: Problem allocating %d bytes.\n", (int)length); SampleSourceClose(&sampleSource); return EXIT_SOFTWARE; }
	samples->buffer = buffer;

	int chans = sampleSource.numChannels;
	if (chans > 3) { chans = 3;	}	// Ignore aux

	// Calcuate SVM of signal
	size_t span;
	const int16_t *data = SampleSourceRead(&sampleSource, 0, samples->numSamples, &span);
	const unsigned char *p = (const unsigned char *)data;

	if (chans == 3 && span == 8 && sampleSource.scale[0] == sampleSource.scale[1] && sampleSource.scale[0] == sampleSource.scale[2] && fabs(sampleSource.scale[0] - (float)(8 / 32768.0)) < 0.001f)
	{
		unsigned int count = samples->numSamples;
		const int16_t *vals = (const int16_t *)p;

		fprintf(stderr, "Calculating SVM (fast path)...\n");
#ifdef WIN_32
#pragma loop(hint_parallel(0))
#endif
		for (unsigned int i = 0; i < count; i++)
		{
			float x = vals[4 * i + 0] * (float)(8 / 32768.0f);
			float y = vals[4 * i + 1] * (float)(8 / 32768.0f);
			float z = vals[4 * i + 2] * (float)(8 / 32768.0f);
			buffer[i] = (float)sqrt(x * x + y * y * z * z);
		}

#ifdef FILTER
		fprintf(stderr, "Creating filter %d...\n", FILTER);
		float circ[FILTER] = { 0 };
		for (unsigned int i = 0; i < count; i++)
		{
			circ[i % FILTER] = buffer[i];
			float sum = 0.0f;
			for (unsigned int j = 0; j < FILTER; j++)
			{
				sum += circ[j];
			}
			buffer[i] = sum / FILTER;
		}
#endif

#ifdef NOISE
		fprintf(stderr, "Adding noise %f...\n", NOISE);
		for (unsigned int i = 0; i < count; i++)
		{
			buffer[i] += NOISE * rand() / RAND_MAX;
		}
#endif

#ifdef DEBUG_OUTPUT
		fprintf(stderr, "Creating debug output...\n");
		for (unsigned int i = 0; i < samples->numSamples; i++)
		{
			// Write over fourth channel
			float v = buffer[i] * 32768.0f / 8;
			if (v <= -32768.0f) { v = -32768.0f; }
			if (v >= 32767.0f) { v = 32767.0f; }
			((int16_t *)vals)[4 * i + 3] = (int16_t)v;
		}
		fprintf(stderr, "Saving debug output...\n");
		char f[256];
		strcpy(f, filename);
		strcat(f, ".svm.wav");
		FILE *fp = fopen(f, "wb");
		if (fp != NULL)
		{
			fwrite(sampleSource.buffer, 1, sampleSource.bufferLength, fp);
			fclose(fp);
		}
#endif

	}
	else
	{
		const unsigned int count = samples->numSamples;
		const int cchans = chans;
		const size_t cspan = span;

		fprintf(stderr, "Calculating SVM (slow path)...\n");
#ifdef WIN_32
#pragma loop(hint_parallel(0))
#endif
		for (unsigned int i = 0; i < count; i++, p += cspan)
		{
			const int16_t *vals = (const int16_t *)p;
			float sum = 0.0f;
			for (int chan = 0; chan < cchans; chan++)
			{
				float v = (float)vals[chan] * sampleSource.scale[chan];
				sum += v * v;
			}
			buffer[i] = (float)sqrt(sum);
		}
	}

	SampleSourceClose(&sampleSource);
	
	return EXIT_OK;
}


// Returns a pointer to the specified sample index for the specified minimum number of samples
const float *SamplesRead(samples_t *samples, unsigned int index, unsigned int minCount)
{
	return (const float *)(samples->buffer + index);
}


// Close the source of samples
void SamplesClose(samples_t *samples)
{
	;
}

