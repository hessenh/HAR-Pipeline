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

#define FILTER		// band-pass filter
//#define BOX_FILTER (1*200/5)
//#define NOISE 0.1f
//#define DEBUG_OUTPUT

#define ABS			// abs(SVM-1) -- much better with

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


#ifdef FILTER
// Where Fc1 = low cut-off frequency, Fc2 = high cut-off frequency, and Fs = sample frequency:
#include "butter.h"
#define FILTER_ORDER 4
#define Fc1 0.5
#define Fc2 10.0
double B[BUTTERWORTH_MAX_COEFFICIENTS(FILTER_ORDER)];
double A[BUTTERWORTH_MAX_COEFFICIENTS(FILTER_ORDER)];
double Z[BUTTERWORTH_MAX_COEFFICIENTS(FILTER_ORDER) - 1];	// filter state
int numCoefficients = 0;
#endif


int SamplesOpen(samples_t *samples, const char *filename)
{
	sample_source_t sampleSource;
	int ret;
	ret = SampleSourceOpen(&sampleSource, filename);
	if (ret != EXIT_OK) { return ret; }

	samples->numSamples = sampleSource.numSamples;
	samples->sampleRate = sampleSource.sampleRate;
	samples->startTime = sampleSource.startTime;

	strcpy(samples->infoArtist, sampleSource.infoArtist);
	strcpy(samples->infoComment, sampleSource.infoComment);
	strcpy(samples->infoDate, sampleSource.infoDate);
	strcpy(samples->infoName, sampleSource.infoName);

	size_t length = sizeof(float) * samples->numSamples;
	float *buffer = (float *)malloc(length);
	if (buffer == NULL) { fprintf(stderr, "ERROR: Problem allocating %d bytes.\n", (int)length); SampleSourceClose(&sampleSource); return EXIT_SOFTWARE; }
	samples->buffer = buffer;

	int chans = sampleSource.numChannels;
	if (chans > 3) { chans = 3; }	// Ignore aux

#ifdef BOX_FILTER
	float circ[BOX_FILTER] = { 0 };
#endif

#ifdef FILTER
	// Prepare filter
	double Fs = sampleSource.sampleRate;
	double W1 = Fc1 / (Fs / 2);
	double W2 = Fc2 / (Fs / 2);
	numCoefficients = CoefficientsButterworth(FILTER_ORDER, W1, W2, B, A);
	memset(Z, 0, sizeof(double) * (numCoefficients - 1));
#endif

	// Chunk over available samples (instead of all at once)
	unsigned int count = 0;
	for (unsigned int offset = 0; offset < samples->numSamples; offset += count)
	{
		// Calcuate SVM of signal
		size_t span = 0;
		const int16_t *data = SampleSourceRead(&sampleSource, offset, 64 * 1024 * 1024 / sampleSource.numChannels / sizeof(int16_t), &count, &span);	// 64 MB chunks
		const unsigned char *p = (const unsigned char *)data;

		// Fast path is 4-channel +/-8g samples
		if (chans == 3 && span == 8 && sampleSource.scale[0] == sampleSource.scale[1] && sampleSource.scale[0] == sampleSource.scale[2] && fabs(sampleSource.scale[0] - (float)(8 / 32768.0)) < 0.001f)
		{
			const int16_t *vals = (const int16_t *)p;

			fprintf(stderr, "Calculating SVM (fast path)...\n");
#ifdef WIN_32
#pragma loop(hint_parallel(0))
#endif
			for (unsigned int i = 0; i < count; i++)
			{
				float x = vals[4 * i + 0] * 8 / 32768.0f;
				float y = vals[4 * i + 1] * 8 / 32768.0f;
				float z = vals[4 * i + 2] * 8 / 32768.0f;
				buffer[offset + i] = (float)sqrt(x * x + y * y + z * z) - 1.0f;
			}

		}
		else if (chans > 2)		// If triaxial data
		{
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
				buffer[offset + i] = (float)sqrt(sum) - 1.0f;
			}
		}
		else                    // Assume audio data
		{
			fprintf(stderr, "Calculating SUM...\n");
			for (unsigned int i = 0; i < count; i++, p += span)
			{
				const int16_t *vals = (const int16_t *)p;
				float sum = 0.0f;
				for (int chan = 0; chan < chans; chan++)
				{
					float v = (float)vals[chan] * sampleSource.scale[chan];
					sum += v;
				}
				buffer[offset + i] = sum / chans;
			}
		}

#ifdef FILTER
		if (chans > 2)
		{
			fprintf(stderr, "Filtering...\n");
			filterf(numCoefficients, B, A, buffer + offset, buffer + offset, count, Z);
		}
#endif

#ifdef BOX_FILTER
		fprintf(stderr, "Creating box filter %d...\n", BOX_FILTER);
		for (unsigned int i = offset + 0; i < offset + count; i++)
		{
			circ[i % BOX_FILTER] = buffer[i];
			float sum = 0.0f;
			for (unsigned int j = 0; j < BOX_FILTER; j++)
			{
				sum += circ[j];
			}
			buffer[i] = sum / BOX_FILTER;
		}
#endif

#ifdef NOISE
		fprintf(stderr, "Adding noise %f...\n", NOISE);
		for (unsigned int i = 0; i < count; i++)
		{
			buffer[offset + i] += NOISE * rand() / RAND_MAX;
		}
#endif

#ifdef ABS
		if (chans > 2)
		{
			fprintf(stderr, "Abs()...\n");
			for (unsigned int i = 0; i < count; i++)
			{
				buffer[offset + i] = (float)fabs(buffer[offset + i]);
			}
		}
#endif
	}

#ifdef DEBUG_OUTPUT
// TODO: Change to work if chunk over samples (instead of all at once)
#pragma message("TODO: Change to work if chunk over samples (instead of all at once)")
	if (chans == 3 && span == 8)
	{
		const int16_t *vals = (const int16_t *)p;
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
	}
	else
	{
		fprintf(stderr, "Cannot create debug output (must be 4-channel accelerometer)...\n");
	}
#endif

	SampleSourceClose(&sampleSource);
	
	return EXIT_OK;
}


// Returns a pointer to the specified sample index for the specified minimum number of samples
const float *SamplesRead(samples_t *samples, unsigned int index, unsigned int minCount)
{
	return (const float *)(samples->buffer + index);
}


// Free the sample data (but retains metadata in this field)
void SamplesFree(samples_t *samples)
{
	if (samples->buffer != NULL) 
	{
		free((float *)samples->buffer);
		samples->buffer = NULL;
	}
}


// Close the source of samples
void SamplesClose(samples_t *samples)
{
	SamplesFree(samples);
	;		// Already closed
}

