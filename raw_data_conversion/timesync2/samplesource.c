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

#define USE_MMAP

#ifdef _WIN32
	#define _CRT_SECURE_NO_WARNINGS
#else
	#define _POSIX_C_SOURCE 200809L 	// strdup()
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifdef USE_MMAP
	#ifdef _WIN32
		// Windows implementation
		#include "mmap-win32.h"
	#else
		#include <sys/mman.h>
	#endif
#endif

#ifdef _WIN32
	#define open _open
	#define close _close
	#define read _read
	#define stat _stat
	#define fstat _fstat
	#define fileno _fileno
	#define strdup _strdup
#else
	#include <unistd.h>
	//#define strdup(_str) strcpy((char *)malloc(strlen(_str) + 1, (_str)))
#endif




#include "samplesource.h"
#include "timestamp.h"
#include "exits.h"
#include "wav.h"


//
// Auxilliary channel format:  ncttttuu vvvvvvvv   
//                             n=0:   data available
//                             n=1:   data not available
//                             c=0:   no data channels clipped
//                             c=1:   some data channels clipped       [reserve nc=11, perhaps for in-data scaling information?]
//                             tttt=0000: metadata:
//                                      uu=00: other comment
//                                      uu=01: 'artist' file metadata
//                                      uu=10: 'title' file metadata
//                                      uu=11: 'comment' file metadata
//                             tttt=0001: sensor - battery (10 bits, u+v)       [reserve tttt=01uu, perhaps for 12-bit battery]
//                             tttt=0010: sensor - light (10 bits, u+v)         [reserve tttt=10uu, perhaps for 12-bit light]
//                             tttt=0011: sensor - temperature (10 bits, u+v)   [reserve tttt=11uu, perhaps for 12-bit temperature]
//
#define WAV_AUX_UNAVAILABLE			0x8000		// u------- --------    Data not available on one or more channels (1/0)
#define WAV_AUX_CLIPPING			0x4000		// -c------ --------    Data clipped on one or more channels (1/0)
#define WAV_AUX_METADATA_OTHER		0x0000		// --000000 vvvvvvvv    Metadata - other comment
#define WAV_AUX_METADATA_ARTIST		0x0100		// --000001 vvvvvvvv    Metadata - artist
#define WAV_AUX_METADATA_TITLE		0x0200		// --000010 vvvvvvvv    Metadata - title
#define WAV_AUX_METADATA_COMMENT	0x0300		// --000011 vvvvvvvv    Metadata - comment
#define WAV_AUX_SENSOR_BATTERY		0x0400		// --0001vv vvvvvvvv    Sensor - battery
#define WAV_AUX_SENSOR_LIGHT		0x0800		// --0010vv vvvvvvvv    Sensor - light
#define WAV_AUX_SENSOR_TEMPERATURE	0x0c00		// --0011vv vvvvvvvv    Sensor - temperature


int SampleSourceOpen(sample_source_t *sampleSource, const char *filename)
{
	int i;
	WavInfo wavInfo = { 0 };

	fprintf(stderr, "SAMPLESOURCE: Loading header: %s\n", filename);
	sampleSource->fp = fopen(filename, "rb");
	if (sampleSource->fp == NULL) { fprintf(stderr, "ERROR: Cannot open WAV file.\n"); return EXIT_NOINPUT; }

	// Default scale
	for (i = 0; i < SAMPLE_SOURCE_CHANNELS_MAX; i++)
	{
		sampleSource->scale[i] = 1.0f;	// (float)(8.0 / 32768.0);
	}

	memset(sampleSource->infoArtist, 0, sizeof(sampleSource->infoArtist));
	memset(sampleSource->infoName, 0, sizeof(sampleSource->infoName));
	memset(sampleSource->infoComment, 0, sizeof(sampleSource->infoComment));
	memset(sampleSource->infoDate, 0, sizeof(sampleSource->infoDate));

	wavInfo.infoArtist = sampleSource->infoArtist;
	wavInfo.infoName = sampleSource->infoName;
	wavInfo.infoComment = sampleSource->infoComment;
	wavInfo.infoDate = sampleSource->infoDate;

	if (!WavRead(&wavInfo, sampleSource->fp)) { fprintf(stderr, "ERROR: Problem reading WAV file format.\n"); fclose(sampleSource->fp); return EXIT_DATAERR; }
	if (wavInfo.bytesPerChannel != 2) { fprintf(stderr, "ERROR: WAV file format not supported (%d bytes/channel, expected 2 = 16-bit).\n", wavInfo.bytesPerChannel); fclose(sampleSource->fp); return EXIT_DATAERR; }
	if (wavInfo.chans < 1 || wavInfo.chans > SAMPLE_SOURCE_CHANNELS_MAX) { fprintf(stderr, "ERROR: WAV file format not supported (%d channels, expected at least 1 and no more than %d).\n", wavInfo.chans, SAMPLE_SOURCE_CHANNELS_MAX); fclose(sampleSource->fp); return EXIT_DATAERR; }
	if (wavInfo.freq < 1) { fprintf(stderr, "ERROR: WAV file format not supported (%d frequency).\n", wavInfo.freq); fclose(sampleSource->fp); return EXIT_DATAERR; }

	// Extract metadata
	char *line;
	#define MAX_FIELDS 32
	//char *artistLines[MAX_FIELDS]; int numArtistLines = 0; char infoArtist = strdup(wavInfo.infoArtist);
	//for (line = strtok(infoArtist, "\n"); line != NULL; line = strtok(NULL, "\n")) { if (numArtistLines < MAX_FIELDS) { artistLines[numArtistLines++] = line; } }
	//char *nameLines[MAX_FIELDS]; int numNameLines = 0; char infoName = strdup(wavInfo.infoName);
	//for (line = strtok(infoName, "\n"); line != NULL; line = strtok(NULL, "\n")) { if (numNameLines < MAX_FIELDS) { nameLines[numNameLines++] = line; } }
	char *commentLines[MAX_FIELDS]; int numCommentLines = 0; char *infoComment = strdup(wavInfo.infoComment);
	for (line = strtok(infoComment, "\n"); line != NULL; line = strtok(NULL, "\n")) { if (numCommentLines < MAX_FIELDS) { commentLines[numCommentLines++] = line; } }

	// Parse headers
	bool parsedTime = false;
	bool parsedScale[SAMPLE_SOURCE_CHANNELS_MAX] = { 0 };
	double startTime = 0;
	for (i = 0; i < numCommentLines; i++)
	{
		if (strncmp(commentLines[i], "Time:", 5) == 0)
		{
			startTime = TimeParse(commentLines[i] + 5);
			fprintf(stderr, "Time: %s\n", TimeString(startTime, NULL));
			if (startTime > 0) { parsedTime = true; }
		}
		else if (strncmp(commentLines[i], "Scale-", 6) == 0 && (commentLines[i][6] >= '1' && commentLines[i][6] <= '9') && commentLines[i][7] == ':')
		{
			int chan = commentLines[i][6] - '1';
			double val = atof(commentLines[i] + 8);
			sampleSource->scale[chan] = (float)(val / 32768.0);
			fprintf(stderr, "Scale-%d: %f (scale[%d] = %f)\n", chan + 1, val, chan, sampleSource->scale[chan]);
			if (sampleSource->scale[chan] > 0) { parsedScale[chan] = true; }
		}
	}

	// Check we parsed the headers we need
	if (!parsedTime) { fprintf(stderr, "WARNING: Didn't successfully parse a 'Time' header (using zero).\n"); }
	for (i = 0; i < 3; i++)
	{
		if (!parsedScale[i]) { fprintf(stderr, "WARNING: Didn't successfully parse a 'Scale-%d' header (using defaults).\n", i + 1); }
	}


	// Copy header information
	sampleSource->dataStartOffset = wavInfo.offset;
	sampleSource->numChannels = wavInfo.chans;
	sampleSource->numSamples = wavInfo.numSamples;
	sampleSource->sampleRate = wavInfo.freq;
	sampleSource->startTime = startTime;
	sampleSource->buffer = NULL;

	// Page size for memory mapped files (also used in the buffer version)
	sampleSource->pageSize = 65536;					// Default 64kB
#ifdef USE_MMAP
	sampleSource->pageSize = sysconf(_SC_PAGE_SIZE);
#endif

	// Data length
	fseek(sampleSource->fp, 0, SEEK_END);
	sampleSource->fileLength = ftell(sampleSource->fp);
	fseek(sampleSource->fp, 0, SEEK_SET);

	// No data read yet
	sampleSource->bufferOffset = 0;
	sampleSource->bufferLength = 0;  
	
	//free(infoArtist);
	//free(infoName);
	free(infoComment);

	return EXIT_OK;
}


// Returns a pointer to the specified sample index for the specified minimum number of samples, and the span/pitch (in bytes) between samples
const int16_t *SampleSourceRead(sample_source_t *sampleSource, unsigned int index, unsigned int minCount, unsigned int *count, size_t *span)
{
	size_t sspan = sizeof(int16_t) * sampleSource->numChannels;

	// Output the span value
	if (span != NULL) { *span = sspan; }

	if (minCount <= 0) { minCount = 1; }	// If not specified, one sample (but will be padded to page size)

	int sampleOffset = (int)(sampleSource->dataStartOffset + (sspan * index));
	int sampleLength = (int)(sspan * minCount);

	// If read requested outside the buffed/mapped area...
	if (sampleOffset < sampleSource->bufferOffset || sampleOffset + sampleLength > sampleSource->bufferOffset + sampleSource->bufferLength)
	{
		// Align offset to a multiple of sampleSource->pageSize
		int residual = sampleOffset % sampleSource->pageSize;
		int offset = sampleOffset - residual;
		int length = sampleLength + residual;

		// Align length to multiple of sampleSource->pageSize
		length = (length + sampleSource->pageSize - 1) / sampleSource->pageSize * sampleSource->pageSize;

		// Minimum length
		int minimumRead = 1 * 1024 * 1024;		// 1MB minimum read
		if (length < minimumRead)
		{
			length = minimumRead;
		}

		// Clamp offset and length
		if (offset + length > sampleSource->fileLength)
		{
			length = sampleSource->fileLength - offset;
		}

#ifdef USE_MMAP
		if (sampleSource->buffer != NULL)
		{
			munmap((void *)sampleSource->buffer, (size_t)sampleSource->bufferLength);
		}
		fprintf(stderr, "SAMPLESOURCE: Mapping %d bytes...\n", (int)length);
		sampleSource->buffer = (unsigned char *)mmap(NULL, length, PROT_READ, MAP_PRIVATE, fileno(sampleSource->fp), offset);
		if (sampleSource->buffer == MAP_FAILED || sampleSource->buffer == NULL) 
		{ 
			fprintf(stderr, "ERROR: Problem mapping %d bytes.\n", (int)length);
			sampleSource->buffer = NULL;
			sampleSource->bufferOffset = 0;
			sampleSource->bufferLength = 0;
			return NULL;
		}
#else
		if (length != sampleSource->bufferLength)
		{
			fprintf(stderr, "SAMPLESOURCE: (Re-)allocating %d bytes...\n", (int)length);
			sampleSource->buffer = (unsigned char *)realloc(sampleSource->buffer, length);
		}
		if (sampleSource->buffer == NULL) 
		{ 
			fprintf(stderr, "ERROR: Problem (re-)allocating %d bytes.\n", (int)length);
			sampleSource->bufferOffset = 0;
			sampleSource->bufferLength = 0;
			return NULL;
		}
		fprintf(stderr, "SAMPLESOURCE: Reading %d bytes...\n", (int)length);
		fseek(sampleSource->fp, offset, SEEK_SET);
		if (fread(sampleSource->buffer, 1, length, sampleSource->fp) != length)
		{ 
			fprintf(stderr, "ERROR: Problem reading %d bytes.\n", (int)length);
			free(sampleSource->buffer);
			sampleSource->buffer = NULL;
			sampleSource->bufferOffset = 0;
			sampleSource->bufferLength = 0;
			return NULL;
		}
#endif

		// Update records
		sampleSource->bufferOffset = offset;
		sampleSource->bufferLength = length;
	}

	// Return count of samples available
	int dataCount = (int)((sampleSource->bufferLength - (sampleOffset - sampleSource->bufferOffset)) / sspan);
	if (count != NULL)
	{
		*count = dataCount;
	}

	// Return data pointer
	const int16_t *data = (const int16_t *)(sampleSource->buffer + sampleOffset - sampleSource->bufferOffset);

	return data;
}


// Close the source of samples
void SampleSourceClose(sample_source_t *sampleSource)
{
	if (sampleSource->buffer != NULL)
	{
#ifdef USE_MMAP
		munmap((void *)sampleSource->buffer, (size_t)sampleSource->bufferLength);
#else
		free((void *)sampleSource->buffer);
#endif
		sampleSource->buffer = NULL;
	}
	sampleSource->bufferLength = 0;
	if (sampleSource->fp != NULL)
	{
		fclose(sampleSource->fp);
		sampleSource->fp = NULL;
	}
}

