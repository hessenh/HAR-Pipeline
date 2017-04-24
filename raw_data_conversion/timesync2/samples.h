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

#ifndef SAMPLES_H
#define SAMPLES_H

#define SAMPLE_MAX_META 16384

// Samples
typedef struct
{
	double startTime;				// seconds since epoch
	const float *buffer;
	unsigned int numSamples;
	unsigned int sampleRate;

	char infoArtist[SAMPLE_MAX_META];
	char infoName[SAMPLE_MAX_META];
	char infoComment[SAMPLE_MAX_META];
	char infoDate[SAMPLE_MAX_META];
} samples_t;


// Open the samples (.WAV file), nonzero is failure
int SamplesOpen(samples_t *samples, const char *filename);

// Free the sample data (but retains metadata)
void SamplesFree(samples_t *samples);

// Close the samples
void SamplesClose(samples_t *samples);

// Returns a pointer to the specified sample index for the specified minimum number of samples
const float *SamplesRead(samples_t *samples, unsigned int index, unsigned int minCount);


#endif
