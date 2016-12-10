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

// Open Movement Time Synchronization
// Dan Jackson

//#define ALT
#define NORMALIZING


#ifdef _WIN32
	#define _CRT_SECURE_NO_WARNINGS
#else
	#define _POSIX_C_SOURCE 200809L 	// strdup()
#endif

/*
Pearson's correlation coefficient:

						   n * SUMi(X[i] * Y[i]) - (SUMi(X[i]) * SUMi(Y[i]))
	r[xy] = ---------------------------------------------------------------------------------
			 sqrt(n * SUMi(X[i]^2) - SUMi(X[i])^2)  *  sqrt(n * SUMi(Y[i]^2) - SUMi(Y[i])^2) 

OR:

						   SUMi(X[i] * Y[i]) - (n * meanX * meanY)
	r[xy] = ---------------------------------------------------------------------
			 sqrt(SUMi(X[i]^2) - n * meanX^2) * sqrt(SUMi(Y[i]^2) - n * meanY^2) 

OR:

			 SUMi(X[i] * Y[i]) - (n * meanX * meanY)
	r[xy] = -----------------------------------------
				 (n - 1) * STDDEV(X) * STDDEV(Y)

where STDDEV(V) = sqrt((1/(n-1)) * SUMi((X[i]-meanX)^2)).

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
#include <math.h>

#include "timesync.h"
#include "timestamp.h"
#include "linearregression.h"
#include "samples.h"
#include "exits.h"
#include "thread.h"
#include "bitmap.h"
#include "samplesource.h"			// Output re interpreted dependent file
#include "wav.h"					// Output re interpreted dependent file

#ifdef _WIN32
	#include "mmap-win32.h"
	#define strdup _strdup
#else
	#include <sys/sysinfo.h>		// get_nprocs()
#endif

#define SLIDING
#define MAX_PROCESS_COUNT 32
//#define LOOP_PARALLEL
typedef double outer_real_t;
typedef float inner_real_t;


#ifdef _WIN32
	#define restrict __restrict
#endif


// Correlation result
typedef struct
{
	int minShift;					// minimum number of samples shifted
	int lengthShift;				// number of sample shift results
	float *correlationCoefficients;	// correlation coefficient results
	float *minStdDev;				// minimum std-dev of master/dependent

	// analysis: simple stats
	float minCoefficient;			// smallest coefficient value
	float maxCoefficient;			// largest coefficient value
	float meanCoefficient;			// mean coefficient value
	int maxIndex;					// index of the largest coefficent (+minShift for shift amount)

	// analysis
	int medianShift;				// local shift amount after median filter (compare with maxIndex+minShift for limit)
	bool useMaximum;				// use this maximum (if it's within range of the median values)
} correlation_result_t;


// Analysis result
typedef struct
{
	bool calculated;				// wether this has been calculated
	bool valid;						// validity of this rate
	int window;						// (centre point) window number
	double offset;					// line of best fit offset (in sample offset shift)
	double scale;					// line of best fit scaling (in sample offset per window size)
	int count;

	// housekeeping
	int startWindow;				// first window
	int endWindow;					// last(+1) window
} rate_t;


// Time sync state
typedef struct
{
	timesync_settings_t *settings;
	samples_t masterSamples;
	samples_t dependentSamples;
	int chans;
	unsigned int sampleRate;
	double dependentTimeOffset;
	int dependentSampleOffset;

	int processCount;

	// 
	int searchLastWindow;			// window number
	double searchOffset;			// Samples
	double searchScale;				// Samples
	double searchInitialSize;		// Seconds
	double searchGrowth;			// Seconds per second

	int numWindows;					// total number of windows
	int windowsComplete;			// 'current' number of windows completed
	int windowsStarted;				// 'current' number of windows where processing has started
	correlation_result_t *correlationResults;

	// analysis
	int outputMinShift;				// overall minimum sample shift searched
	int outputMaxShift;				// overall maximum sample shift searched
	int numRates;					// number of rates
	int numValid;					// number of valid rates
	rate_t *rates;					// output rates
	double *bestFitShift;			// Overall best fit shift amount for each window
} timesync_t;


// Initialize the settings to default values
void TimeSyncDefaults(timesync_settings_t *settings)
{
	memset(settings, 0, sizeof(timesync_settings_t));

	settings->windowSizeTime = 60.0;						// 1-minute windows
	settings->windowSkipTime = settings->windowSizeTime;

	settings->analysisSkipTime = 24 * 60.0 * 60.0;			// 24 hours

	settings->processCount = -1;							// (<=0 auto plus abs)

	settings->firstSearchInitialSize = 20.0f;
	settings->firstSearchGrowth = 6.0f / (24 * 60 * 60);	// 6 seconds in 24 hours

	settings->searchInitialSize = 2.0f;
	settings->searchGrowth = 6.0f / (24 * 60 * 60);			// 6 seconds in 24 hours


	settings->medianWindowTime = 0.1;
	settings->minStdDev = 0.0;
	settings->medianMaxTime = 0.05;							// 0.1 s ?

	settings->analysisMinimumCount = 10;					// At least 10 points
	settings->analysisMaxTime = 0.1;						// Within 0.1 seconds of the line
	settings->analysisMinimumProportion = 0.3;				// 30% must be on the line

	settings->csvSkip = 1;
}


// Open the time sync object
int TimeSyncOpen(timesync_t *timesync, timesync_settings_t *settings)
{
	memset(timesync, 0, sizeof(timesync_t));
	timesync->settings = settings;

	// Master file
	int masterOk = SamplesOpen(&timesync->masterSamples, settings->masterFilename);
	if (masterOk != EXIT_OK)
	{
		fprintf(stderr, "ERROR: Problem opening master.\n");
		return masterOk;
	}

	// Dependent file
	int dependentOk = SamplesOpen(&timesync->dependentSamples, settings->dependentFilename);
	if (dependentOk != EXIT_OK)
	{
		fprintf(stderr, "ERROR: Problem opening dependent.\n");
		SamplesClose(&timesync->masterSamples);
		return dependentOk;
	}

	// Sample rate must match
	if (timesync->masterSamples.sampleRate != timesync->dependentSamples.sampleRate)
	{
		fprintf(stderr, "ERROR: Master and dependent frequencies mismatch (%d Hz vs. %d Hz).\n", timesync->masterSamples.sampleRate, timesync->dependentSamples.sampleRate);
		SamplesClose(&timesync->masterSamples);
		SamplesClose(&timesync->dependentSamples);
		return EXIT_CONFIG;
	}


	return EXIT_OK;
}


// Close the time sync object
int TimeSyncClose(timesync_t *timesync)
{
	SamplesClose(&timesync->masterSamples);
	SamplesClose(&timesync->dependentSamples);
	return EXIT_OK;
}


static outer_real_t mean(const float *restrict v, const int n, outer_real_t *outVarianceSum)
{
	int i;

	// Calculate the mean of X and SX
	outer_real_t meanSum = 0.0f;

#if defined(_WIN32) && defined(LOOP_PARALLEL)
#pragma loop(hint_parallel(0))
#endif
	for (i = 0; i < n; i++)
	{
		meanSum += v[i];
	}
	outer_real_t mean = meanSum / n;

	outer_real_t varianceSum = 0.0f;
#if defined(_WIN32) && defined(LOOP_PARALLEL)
#pragma loop(hint_parallel(0))
#endif
	for (i = 0; i < n; i++)
	{
		varianceSum += (v[i] - mean) * (v[i] - mean);
	}

	*outVarianceSum = varianceSum;

	return mean;
}

static float correlation(const float *restrict x, const float *restrict y, const int n, const outer_real_t outerMeanX, const outer_real_t outerMeanY, const outer_real_t sx, const outer_real_t sy, float *minStdDev)
{
	const inner_real_t meanX = (const inner_real_t)outerMeanX;
	const inner_real_t meanY = (const inner_real_t)outerMeanY;
	int i;

	// Calculate the correlation
	inner_real_t sxy = 0.0f;

#if defined(_WIN32) && defined(LOOP_PARALLEL)
#pragma loop(hint_parallel(0))
#endif
	for (i = 0; i < n; i++)
	{
#ifdef ALT
		sxy += x[i] * y[i];
#else
		sxy += (x[i] - meanX) * (y[i] - meanY);
#endif
	}

#ifdef ALT
	sxy -= n * meanX * meanY;
	double stddevX = sqrt((1.0 / (n - 1)) * sx);
	double stddevY = sqrt((1.0 / (n - 1)) * sy);
	double denom = (n - 1) * stddevX * stddevY;
#else
	// Calculate the denominator
	double denom = sqrt((double)sx * (double)sy);
#endif
	if (denom == 0.0f) { denom = 1.0f; }
	float correlationCoefficient = (float)(sxy / denom);

	*minStdDev = (float)fmin(sqrt((1.0 / (n - 1)) * sx), sqrt((1.0 / (n - 1)) * sy));

	return correlationCoefficient;
}


// Correlation at a given time
void TimeSyncCorrelate(timesync_t *timesync, double masterOffsetTime, double minShiftTime, double maxShiftTime, double windowSizeTime, correlation_result_t *correlationResult)
{
	// Zero
	memset(correlationResult, 0, sizeof(correlation_result_t));

	// Calculate first and (one after) last master sample for the window
	int masterStart = (int)((masterOffsetTime - (windowSizeTime / 2)) * timesync->masterSamples.sampleRate);
	int masterEnd = (int)((masterOffsetTime + (windowSizeTime / 2)) * timesync->masterSamples.sampleRate);
	if (masterStart < 0) { masterStart = 0; }
	if (masterEnd > (int)timesync->masterSamples.numSamples) { masterEnd = timesync->masterSamples.numSamples; }
	int masterLength = masterEnd - masterStart;
	if (masterLength <= 0)
	{
		//fprintf(stderr, "WARNING: No master samples at time %f\n", masterOffsetTime);
		fprintf(stderr, "!");
		return;
	}

	// Calculate shift samples
	int minShift = (int)(minShiftTime * timesync->dependentSamples.sampleRate);
	int maxShift = (int)(maxShiftTime * timesync->dependentSamples.sampleRate);

	// Calculate first and (one after) last dependent sample for the window + shift
	int dependentStart = masterStart - timesync->dependentSampleOffset + minShift;
	int dependentEnd = masterEnd - timesync->dependentSampleOffset + maxShift;
	if (dependentStart < 0)
	{
		minShift += -dependentStart;
		dependentStart = 0;
	}
	if (dependentEnd > (int)timesync->dependentSamples.numSamples)
	{
		maxShift -= dependentEnd - timesync->dependentSamples.numSamples;
		dependentEnd = timesync->dependentSamples.numSamples;
	}

	// Check interval
	if (dependentEnd <= dependentStart || maxShift < minShift)
	{
		//fprintf(stderr, "WARNING: No dependent samples to compare.\n");
		fprintf(stderr, "?");
		return;
	}

	// 
	int dependentLength = dependentEnd - dependentStart;
	int lengthShift = maxShift - minShift;
	if (dependentLength < masterLength)
	{
		fprintf(stderr, "WARNING: Trimming master length for available dependent samples.\n");
		masterLength = dependentLength;
		lengthShift = 0;
	}

	// Read samples
	const float *masterSamples = SamplesRead(&timesync->masterSamples, masterStart, masterLength);
	const float *dependentSamples = SamplesRead(&timesync->dependentSamples, dependentStart, dependentLength);

//	fprintf(stderr, "TIMESYNC: @%.3f +(%.3f,%.3f) [%.3f]\n", masterOffsetTime, minShiftTime, maxShiftTime, windowSizeTime);
//	fprintf(stderr, "--> @%d [%d] @%d +%d +%d =%d\n", masterStart, masterLength, dependentStart, masterLength, lengthShift, dependentEnd);

	if (dependentStart + masterLength + lengthShift != dependentEnd)
	{
		fprintf(stderr, "WARNING: dependentStart + masterLength + lengthShift != dependentEnd\n");
	}

	// Cross-correlate
	const int n = masterLength;
	int i;

	outer_real_t sx;
	outer_real_t meanX = mean(masterSamples, n, &sx);
	
#ifdef SLIDING
	// Sliding mean of Y variance-sum of Y
	const float *y = dependentSamples + 0;
	double meanSum = 0.0f;
#if defined(_WIN32) && defined(LOOP_PARALLEL)
#pragma loop(hint_parallel(0))
#endif
	for (i = 0; i < n; i++)
	{
		meanSum += y[i];
	}
	double meanY = meanSum / n;
	double varianceSum = 0.0f;
#if defined(_WIN32) && defined(LOOP_PARALLEL)
#pragma loop(hint_parallel(0))
#endif
	for (i = 0; i < n; i++)
	{
		varianceSum += (y[i] - meanY) * (y[i] - meanY);
	}
#endif

	// Prepare results
	correlationResult->minShift = minShift;
	correlationResult->lengthShift = lengthShift;
	correlationResult->correlationCoefficients = calloc(correlationResult->lengthShift, sizeof(float));
	correlationResult->minStdDev = calloc(correlationResult->lengthShift, sizeof(float));

	// Process
	for (i = 0; i < lengthShift; i++)
	{
#ifdef SLIDING
		// Move Y along by 1
		y = dependentSamples + i;

		if (i > 0)
		{
			// Recalculate mean Y and variance-sum of Y
			float y_old = y[-1];
			float y_new = y[n - 1];
			meanSum = meanSum - y_old + y_new;
			double old_meanY = meanY;
			meanY = meanSum / n;
			varianceSum += (y_new - old_meanY) * (y_new - meanY) - (y_old - old_meanY) * (y_old - meanY);
		}
		outer_real_t sy = (outer_real_t)varianceSum;
#else
		const float *y = dependentSamples + i;
		outer_real_t sy;
		outer_real_t meanY = mean(y, n, &sy);
#endif

		float minStdDev;
		float correlationCoefficient = correlation(masterSamples, y, n, meanX, meanY, sx, sy, &minStdDev);
		correlationResult->correlationCoefficients[i] = correlationCoefficient;
		correlationResult->minStdDev[i] = minStdDev;
		

//		printf("%d,%.3f,%.3f\n", i, time, correlationCoefficient);

	}

}


// Find the best correlation coefficient and time
/*
static float BestCorrelation(timesync_t *timesync, const correlation_result_t *correlationResult, float *outBestCoefficient)
{
	float bestCorrelation = -2.0f;
	float bestCorrelationTime = 0.0f;

	if (correlationResult->correlationCoefficients == NULL)
	{
		*outBestCoefficient = INFINITY;
		return INFINITY;
	}

	for (int i = 0; i < correlationResult->lengthShift; i++)
	{
		float correlationCoefficient = correlationResult->correlationCoefficients[i];
		if (correlationCoefficient > bestCorrelation)
		{
			float time = (float)(correlationResult->minShift + i) / timesync->dependentSamples.sampleRate;
			bestCorrelation = correlationCoefficient;
			bestCorrelationTime = time;
		}
	}

	*outBestCoefficient = bestCorrelation;
	return bestCorrelationTime;
}
*/


// Thread-local configuration and results
typedef struct
{
	int name;
	volatile bool active;
	volatile bool discard;	// throw away result (settings have changed for this window, must recompute)

	volatile int i;

	timesync_t *timesync;
	volatile double masterOffsetTime;
	volatile double minShiftTime;
	volatile double maxShiftTime;
	volatile double windowSizeTime;

	correlation_result_t correlationResult;

	event_t eventStart;
	event_t *eventFinish;				// shared
	volatile bool quit;
} process_data_t;



unsigned int THREAD_CALL Process(process_data_t *processData)
{
	//fprintf(stderr, "THREAD#%d:start\n", processData->name);
	while (!processData->quit)
	{
//fprintf(stderr, "T%d:wait-start-%d\n", processData->name, processData->name);
		event_wait(&processData->eventStart);
//fprintf(stderr, "T%d:~wait-start-%d @%d\n", processData->name, processData->name, processData->i);
		if (processData->quit) { break; }

		TimeSyncCorrelate(processData->timesync, processData->masterOffsetTime, processData->minShiftTime, processData->maxShiftTime, processData->windowSizeTime, &processData->correlationResult);

		processData->active = false;

//fprintf(stderr, "T%d:signal-finish\n", processData->name);
		event_signal(processData->eventFinish);
	}
	//fprintf(stderr, "THREAD#%d:end\n", processData->name);
	return 0;
}

/*
static int float_compare(const void *a, const void *b)
{
	float fa = *(const float *)a;
	float fb = *(const float *)b;
	return (fa > fb) - (fa < fb);
}
*/

static int int_compare(const void *a, const void *b)
{
	int fa = *(const int *)a;
	int fb = *(const int *)b;
	return (fa > fb) - (fa < fb);
}


// Periodically analyse the output
static bool TimeSyncAnalyse(timesync_t *timesync)
{
	// Should we analyse the rate?
	int rateIndex = (timesync->numRates * timesync->windowsComplete / timesync->numWindows) - 1;
	if (rateIndex < 0 || timesync->rates[rateIndex].calculated)
	{
		return false;
	}

	// Analysis window
	rate_t *rate = &timesync->rates[rateIndex];
	rate->calculated = true;
	rate->startWindow = 0;
	rate->endWindow = timesync->windowsComplete;
	if (rateIndex > 0)
	{
		rate->startWindow = timesync->rates[rateIndex - 1].endWindow;
	}
	rate->window = (rate->startWindow + rate->endWindow) / 2;


	fprintf(stderr, "Analysing @%d-%d (~%d) / %d...\n", rate->startWindow, rate->endWindow, rate->window, timesync->numWindows);

	// Calculate overall shift range
	//timesync->outputMinShift = 0;
	//timesync->outputMaxShift = 0;
	for (int i = rate->startWindow; i < rate->endWindow; i++)
	{
		bool first = (rateIndex == 0) && (i == rate->startWindow);
		if (timesync->correlationResults[i].correlationCoefficients == NULL) { continue; }
		if (first || timesync->correlationResults[i].minShift < timesync->outputMinShift)
		{
			timesync->outputMinShift = timesync->correlationResults[i].minShift;
		}
		if (first || timesync->correlationResults[i].minShift + timesync->correlationResults[i].lengthShift > timesync->outputMaxShift)
		{
			timesync->outputMaxShift = timesync->correlationResults[i].minShift + timesync->correlationResults[i].lengthShift;
		}
	}

	// Simple stats
	for (int window = rate->startWindow; window < rate->endWindow; window++)
	{
		correlation_result_t *results = &timesync->correlationResults[window];

		// Calculate min/max/mean
		float min = 0, sum = 0, max = 0;
		int maxIndex = -1;
		for (int i = 0; i < results->lengthShift; i++)
		{
			float v = results->correlationCoefficients[i];
			sum += v;
			if (i == 0 || v < min) { min = v; }
			if (i == 0 || v > max) { max = v; maxIndex = i; }
		}
		results->minCoefficient = min;
		results->meanCoefficient = sum / results->lengthShift;
		results->maxCoefficient = max;
		results->maxIndex = maxIndex;
	}

	// Median filter
	int medianSize = (int)(timesync->settings->medianWindowTime * timesync->sampleRate);
	int *median = (int *)malloc(medianSize * sizeof(int));
	for (int window = rate->startWindow; window < rate->endWindow; window++)
	{
		// Calculate the median shift amount
		int n = 0;
		for (int i = 0; i < medianSize; i++)
		{
			int j = window - (medianSize / 2) + i;
			if (j >= window) { j++; }		// Do not include the current window in the median calculation
			if (j >= 0 && j < rate->endWindow && timesync->correlationResults[j].maxIndex >= 0)
			{
				median[n++] = timesync->correlationResults[j].maxIndex + timesync->correlationResults[j].minShift;
			}
		}
		qsort(median, n, sizeof(median[0]), int_compare);

		if (n > 0)
		{
			timesync->correlationResults[window].medianShift = median[n / 2];
			int medianDistance = ((int)(timesync->sampleRate * timesync->settings->medianMaxTime));	// 0.1 seconds
			float minStdDev = -1.0f;
			if (timesync->correlationResults[window].minStdDev != NULL && timesync->correlationResults[window].maxIndex >= 0)
			{
				minStdDev = timesync->correlationResults[window].minStdDev[timesync->correlationResults[window].maxIndex];
			}

			if (minStdDev >= timesync->settings->minStdDev)
			{
				if (medianDistance > 0)
				{
					timesync->correlationResults[window].useMaximum = abs(timesync->correlationResults[window].medianShift - (timesync->correlationResults[window].maxIndex + timesync->correlationResults[window].minShift)) < medianDistance;
				}
				else
				{
					timesync->correlationResults[window].useMaximum = true;
				}
			}
		}
		else
		{
			timesync->correlationResults[window].medianShift = INT_MIN;
		}
	}
	free(median);


	// Count number of points to use for linear regression
	rate->count = 0;
	for (int window = rate->startWindow; window < rate->endWindow; window++)
	{
		if (timesync->correlationResults[window].useMaximum)
		{
			rate->count++;
		}
	}
	double *dataX = (double *)calloc(rate->count, sizeof(double));
	double *dataY = (double *)calloc(rate->count, sizeof(double));
	int i = 0;
	for (int window = rate->startWindow; window < rate->endWindow; window++)
	{
		if (timesync->correlationResults[window].useMaximum)
		{
			int shift = timesync->correlationResults[window].maxIndex + timesync->correlationResults[window].minShift;

			dataX[i] = (double)window;
			dataY[i] = shift;

			//printf("%f,%f\n", dataX[i], dataY[i]);

			i++;
		}
	}
	double *fit = LinearModelFitOneIndependent(rate->count, dataY, dataX);
	free(dataX);
	free(dataY);

	// Save global best fit adjusted for mid-point
	// Line of best fit: y = scale * x + offset
	//fprintf(stderr, "Line of best fit:  y = scale * x + offset  (offset = sample offset, x in windows of 60 seconds)\n");
	rate->offset = fit[0];
	rate->scale = fit[1];

	// Check validity
	int withinDistance = 0;
	i = 0;
	for (int window = rate->startWindow; window < rate->endWindow; window++)
	{
		if (timesync->correlationResults[window].useMaximum)
		{
			int shift = timesync->correlationResults[window].maxIndex + timesync->correlationResults[window].minShift;

			double y = rate->offset + rate->scale * window;
			double diff = fabs(y - shift);

			int maxDistance = ((int)(timesync->sampleRate * timesync->settings->analysisMaxTime));	// 0.1 seconds
			if (maxDistance <= 0.0 || diff <= maxDistance)
			{
				withinDistance++;
			}
		}
	}

	rate->valid = (rate->count > timesync->settings->analysisMinimumCount && withinDistance >= (int)(timesync->settings->analysisMinimumProportion * rate->count));

	fprintf(stderr, "RATE #%d/%d (%d points) @%d: %f %f %s\n",
		rateIndex + 1, timesync->numRates, rate->count,
		rate->window, rate->offset, rate->scale, rate->valid ? "VALID" : "INVALID");


	if (rate->valid)
	{
		timesync->searchLastWindow = timesync->windowsComplete;
		timesync->searchOffset = rate->offset;
		timesync->searchScale = rate->scale;

		timesync->searchInitialSize = timesync->settings->searchInitialSize;
		timesync->searchGrowth = timesync->settings->searchGrowth;

		timesync->numValid++;
	}

	fprintf(stderr, "Adjusting search interval (%s) at %f size %f\n",
		rate->valid ? "VALID" : "INVALID", (timesync->searchOffset + timesync->searchScale * timesync->searchLastWindow) / timesync->sampleRate, timesync->searchInitialSize);

	return true;
}


static bool TimeSyncOutputImage(timesync_t *timesync)
{
	// Output image
	if (timesync->settings->imageFilename == NULL)
	{
		return false;
	}

	// Calculate dimensions
	int height = timesync->outputMaxShift - timesync->outputMinShift;
	int width = timesync->numWindows;
	int span = (3 * width + 3) / 4 * 4;

	// Output image
	fprintf(stderr, "Writing BMP file: %s\n", timesync->settings->imageFilename);
	FILE *fp = fopen(timesync->settings->imageFilename, "wb");
	if (fp == NULL)
	{
		fprintf(stderr, "ERROR: Problem writing BMP file: %s\n", timesync->settings->imageFilename);
		return false;
	}

	// Output bitmap header
	unsigned char headerBuffer[BMP_WRITER_SIZE_HEADER];
	BitmapHeader(headerBuffer, NULL, width, height, 24);
	fwrite(headerBuffer, 1, sizeof(headerBuffer), fp);
	unsigned char *buffer = (unsigned char *)calloc(span, sizeof(unsigned char));

	for (int y = 0; y < height; y++)
	{
		int shift = y + timesync->outputMinShift;

		for (int x = 0; x < width; x++)
		{
			int bestY = (int)timesync->bestFitShift[x] - timesync->outputMinShift;
			bool peak = false, allowed = false;
			//bool median = false;
			bool best = (y == bestY);

			// Calculate coefficient for this time
			float fv = INFINITY;
			int window = x;
			if (window < timesync->windowsComplete)
			{
				float *coefficients = timesync->correlationResults[window].correlationCoefficients;
				if (coefficients != NULL)
				{
					if (shift >= timesync->correlationResults[window].minShift && shift < timesync->correlationResults[window].minShift + timesync->correlationResults[window].lengthShift)
					{
						int si = shift - timesync->correlationResults[window].minShift;
						fv = coefficients[si];
#ifdef NORMALIZING

#if 0
						// 'sharpen'
						if (si > 50 && si < timesync->correlationResults[window].lengthShift - 50)
						{
							fv = (coefficients[si] - coefficients[si - 50]) + (coefficients[si] - coefficients[si + 50]);
						}
						fv = fv * 2.0f - 1.0f;
#endif

						//fv = (fv - normalizingMean[x]) * 2.0f;	// mean
						//fv = (fv - normalizingMin[x]) * 2.0f - 1.0f;	// min

						fv = (fv - timesync->correlationResults[window].minCoefficient) * 2.0f - 1.0f;	// min
						int maxY = timesync->correlationResults[window].maxIndex + timesync->correlationResults[window].minShift - timesync->outputMinShift;
						if (y == maxY)
						{
							fv = 1.0f; peak = true;
							allowed = timesync->correlationResults[window].useMaximum;
						}

						if (fv > 1.0f) { fv = 1.0f; }
						if (fv < -1.0f) { fv = -1.0f; }
#endif
					}
				}

				//if (y == timesync->correlationResults[window].medianShift) { median = true; }
			}

			// Calculate output value
			unsigned char r, g, b;
			if (fv >= -1.0f && fv <= 1.0f)
			{
				// Blue
				float vv = (float)((0.5f * fv + 0.5f) * 255.0f);

				// Saturate
				unsigned char v;
				if (vv >= 255.0f) { v = 0xff; }
				else if (vv <= 0.0f) { v = 0x00; }
				else { v = (unsigned char)vv; }

				if (peak)
				{
					if (allowed)
					{
						r = 0xff; g = 0xff; b = 0xff;
					}
					else
					{
						r = 0x80; g = 0x40; b = 0x40;
					}
				}
				else
				{
					b = v / 2;
					r = g = (b / 4);
				}
			}
			else
			{
				r = g = b = 0x33;
			}

			if (best)
			{
				g = 0xcc;
			}

			// Output BGR
			unsigned char *p = buffer + (3 * x);
			p[0] = b; p[1] = g; p[2] = r;

		}
		fwrite(buffer, 1, span, fp);
	}

	free(buffer);

	fclose(fp);

	return true;
}


static bool TimeSyncOutputCsv(timesync_t *timesync)
{
	if (timesync->settings->csvFilename == NULL)
	{
		return false;
	}

	int ret;

	// master file
	fprintf(stderr, "Loading master file to output to CSV: %s\n", timesync->settings->masterFilename);
	sample_source_t sampleSourceMaster;
	ret = SampleSourceOpen(&sampleSourceMaster, timesync->settings->masterFilename);
	if (ret != EXIT_OK) { fprintf(stderr, "ERROR: Failed to open master: %s\n", timesync->settings->masterFilename); return ret; }
	size_t masterSpan;
	const int16_t *masterData = SampleSourceRead(&sampleSourceMaster, 0, sampleSourceMaster.numSamples, &masterSpan);

	// dependent file
	fprintf(stderr, "Loading dependent file to resample to CSV: %s\n", timesync->settings->dependentFilename);
	sample_source_t sampleSourceDependent;
	ret = SampleSourceOpen(&sampleSourceDependent, timesync->settings->dependentFilename);
	if (ret != EXIT_OK) { fprintf(stderr, "ERROR: Failed to open dependent: %s\n", timesync->settings->dependentFilename); return ret; }
	size_t dependentSpan;
	const int16_t *dependentData = SampleSourceRead(&sampleSourceDependent, 0, sampleSourceDependent.numSamples, &dependentSpan);

	// Output file
	fprintf(stderr, "Creating resampled CSV file: %s\n", timesync->settings->csvFilename);
	FILE *fp = fopen(timesync->settings->csvFilename, "wb");
	if (fp == NULL)
	{
		fprintf(stderr, "ERROR: Problem writing CSV file: %s\n", timesync->settings->csvFilename);
		return false;
	}

	// Start/end times
	int startX = 0;
	int endX = sampleSourceMaster.numSamples;
	if (timesync->settings->csvStartTime != 0.0)
	{
		fprintf(stderr, "CSV requested start time: %s\n", TimeString(timesync->settings->csvStartTime, NULL));
		startX = (int)((timesync->settings->csvStartTime - sampleSourceMaster.startTime) * sampleSourceMaster.sampleRate);
		if (startX < 0) { fprintf(stderr, "WARNING: Advancing start time to fit master data\n"); startX = 0; }
	}
	if (timesync->settings->csvEndTime != 0.0)
	{
		fprintf(stderr, "CSV requested end time: %s\n", TimeString(timesync->settings->csvEndTime, NULL));
		endX = (int)((timesync->settings->csvEndTime - sampleSourceMaster.startTime) * sampleSourceMaster.sampleRate);
		if (endX > (int)sampleSourceMaster.numSamples) { fprintf(stderr, "WARNING: Bringing back end time to fit master data\n"); endX = sampleSourceMaster.numSamples; }
	}
	if (startX >= endX) { fprintf(stderr, "WARNING: No output time interval.\n"); }

	// Generate output
	int16_t empty[] = { 0, 0, 0 };
	for (int x = startX; x < endX; x += timesync->settings->csvSkip)
	{
		int window = (int)(x / timesync->settings->windowSkipTime / timesync->sampleRate);
		if (window >= timesync->numWindows) { window = timesync->numWindows - 1; }
		int shift = (int)timesync->bestFitShift[window];

		int src = x + shift;		// ??? -shift
		const int16_t *masterSample = masterData + sampleSourceMaster.numChannels * x;
		const int16_t *dependentSample = empty;
		if (src >= 0 && src < (int)sampleSourceDependent.numSamples)
		{
			dependentSample = dependentData + sampleSourceDependent.numChannels * src;
		}

		fprintf(fp, "%s,%f,%f,%f,%f,%f,%f\n", 
			TimeString(sampleSourceMaster.startTime + (double)x / sampleSourceMaster.sampleRate, NULL), 
			masterSample[0] * sampleSourceMaster.scale[0],
			masterSample[1] * sampleSourceMaster.scale[1],
			masterSample[2] * sampleSourceMaster.scale[2],
			dependentSample[0] * sampleSourceDependent.scale[0],
			dependentSample[1] * sampleSourceDependent.scale[1],
			dependentSample[2] * sampleSourceDependent.scale[2]
			);
	};
	fclose(fp);
	SampleSourceClose(&sampleSourceMaster);
	SampleSourceClose(&sampleSourceDependent);

	return true;
}


static bool TimeSyncOutputResampled(timesync_t *timesync)
{
	if (timesync->settings->outFilename == NULL)
	{
		return false;
	}

	// Source is dependent file
	fprintf(stderr, "Loading file to resample: %s\n", timesync->settings->dependentFilename);
	sample_source_t sampleSource;
	int ret;
	ret = SampleSourceOpen(&sampleSource, timesync->settings->dependentFilename);
	if (ret != EXIT_OK) { fprintf(stderr, "ERROR: Failed to open dependent: %s\n", timesync->settings->dependentFilename); return ret; }
	size_t span;
	const int16_t *data = SampleSourceRead(&sampleSource, 0, sampleSource.numSamples, &span);

	// Output file
	fprintf(stderr, "Creating resampled file: %s\n", timesync->settings->outFilename);
	FILE *fp = fopen(timesync->settings->outFilename, "wb");
	if (fp == NULL)
	{
		fprintf(stderr, "ERROR: Problem writing WAV file: %s\n", timesync->settings->outFilename);
		return false;
	}

	// Modify the comment
	char *oldInfoComment = strdup(timesync->dependentSamples.infoComment != NULL ? timesync->dependentSamples.infoComment : "");	// From original data ("Time: YYYY-MM-DD hh:mm:ss.000")
	char *newInfoComment = malloc(strlen(oldInfoComment) + 128);
	newInfoComment[0] = '\0';
	for (char *line = strtok(oldInfoComment, "\n"); line != NULL; line = strtok(NULL, "\n"))
	{
		if (memcmp(line, "Time:", 5) == 0)
		{
			sprintf(newInfoComment + strlen(newInfoComment), "Time: %s\n", timesync->masterSamples.infoDate);
		}
		else
		{
			strcat(newInfoComment, line); strcat(newInfoComment, "\n");
		}
	}
	free(oldInfoComment);


	// WAVE header
	WavInfo wavInfo = { 0 };
	wavInfo.bytesPerChannel = sizeof(short);						// 2
	wavInfo.chans = sampleSource.numChannels;
	wavInfo.freq = sampleSource.sampleRate;
	wavInfo.numSamples = timesync->masterSamples.numSamples;		// Same output as master
	wavInfo.infoArtist = timesync->dependentSamples.infoArtist;		// From original data
	wavInfo.infoName = timesync->dependentSamples.infoName;			// From original data
	wavInfo.infoComment = newInfoComment;
	wavInfo.infoDate = timesync->masterSamples.infoDate;			// The new start date is from the master track
	wavInfo.offset = 1024;											// Start data at 1k boundary (insert dummy JUNK header)
	WavWrite(&wavInfo, fp);
	free(newInfoComment);


	// Output 
	size_t bufferLength = (size_t)(timesync->settings->windowSkipTime * timesync->sampleRate + 1);
	short *buffer = (short *)malloc(bufferLength * wavInfo.chans * sizeof(short));
	if (buffer == NULL) { fprintf(stderr, "ERROR: Problem allocating %d samples.\n", (int)bufferLength); return EXIT_SOFTWARE; }

	// Generate output
	int x = 0;
	while ((unsigned long)x < wavInfo.numSamples)
	{
		int window = (int)(x / timesync->settings->windowSkipTime / timesync->sampleRate);
		if (window >= timesync->numWindows) { window = timesync->numWindows - 1; }
		int shift = (int)timesync->bestFitShift[window];

		int j = 0;
		for (j = 0; (size_t)j < bufferLength && x < (int)wavInfo.numSamples; j++, x++)
		{
			int src = x + shift;		// ??? -shift
										// Copy sample
			if (src >= 0 && src < (int)sampleSource.numSamples)
			{
				memcpy(&buffer[wavInfo.chans * j], &data[wavInfo.chans * src], span);
			}
			else
			{
				memset(&buffer[wavInfo.chans * j], 0, span);
				if (wavInfo.chans > 3)
				{
					unsigned short *aux = (unsigned short *)&buffer[wavInfo.chans * j + (wavInfo.chans - 1)];
					// Set invalid bit in aux channel
					*aux |= 0x8000;
				}
			}
		}

		fwrite(buffer, sizeof(short) * wavInfo.chans, j, fp);
	};
	free(buffer);
	fclose(fp);
	SampleSourceClose(&sampleSource);


	return true;
}


static void TimeSyncCalcOverallBestFit(timesync_t *timesync)
{
	// What was the best fit value for this window?
	// Line of best fit: y = scale * x + offset
	fprintf(stderr, "Best fit... (%d points => %d windows)\n", timesync->numRates, timesync->numWindows);
	timesync->bestFitShift = (double *)calloc(timesync->numWindows, sizeof(double));
	for (int x = 0; x < timesync->numWindows; x++)
	{
		int lastBefore = -1;
		int firstAfter = -1;
		for (int i = 0; i < timesync->numRates; i++)
		{
			if (timesync->rates[i].valid)
			{
				if (timesync->rates[i].window < x)
				{
					lastBefore = i;
				}
				if (timesync->rates[i].window >= x && firstAfter < 0)
				{
					firstAfter = i;
				}
			}
		}

		// Allow extrapolation
		if (lastBefore < 0) { lastBefore = firstAfter; }
		if (firstAfter < 0) { firstAfter = lastBefore; }

		// Unknown
		if (lastBefore < 0 || firstAfter < 0)
		{
			timesync->bestFitShift[x] = 0;		// unknown drift
		}
		else
		{
			// Evaluate using both best-fit lines
			rate_t *r0 = &timesync->rates[lastBefore];
			rate_t *r1 = &timesync->rates[firstAfter];
			double y0 = r0->offset + x * r0->scale;
			double y1 = r1->offset + x * r1->scale;

			// Calculate proportion between them
			double prop;
			if (x < r0->window)
			{
				prop = 0.0;
			}
			else if (x > r1->window)
			{
				prop = 1.0;
			}
			else if (r0->window == r1->window)
			{
				prop = 0.0;
			}
			else
			{
				prop = (double)(x - r0->window) / (r1->window - r0->window);
			}

			//fprintf(stdout, "X=%d/%d, R0[#%d/%d]@%d, R1[#%d/%d]@%d ==> prop=%f\n", x, timesync->numWindows, lastBefore, timesync->numRates, r0->window, firstAfter, timesync->numRates, r1->window, prop);

			// Apply
			timesync->bestFitShift[x] = (y0 * (1.0 - prop)) + (y1 * prop);
		}
	}

}


// Process the time sync object
int TimeSyncProcess(timesync_t *timesync)
{
	timesync->sampleRate = timesync->masterSamples.sampleRate;
	double masterDuration = (double)timesync->masterSamples.numSamples / timesync->sampleRate;
	fprintf(stderr, "NOTE: %d samples at %d Hz = %f seconds\n", timesync->masterSamples.numSamples, timesync->sampleRate, masterDuration);
	timesync->numWindows = (int)(masterDuration / timesync->settings->windowSkipTime);
	timesync->correlationResults = (correlation_result_t *)calloc(timesync->numWindows, sizeof(correlation_result_t));

	timesync->numRates = (int)(masterDuration / timesync->settings->analysisSkipTime);
	if (timesync->numRates < 1)
	{
		timesync->numRates = 1;
	}
	timesync->rates = (rate_t *)calloc(timesync->numRates, sizeof(rate_t));

	// Initial values
	timesync->searchLastWindow = 0;
	timesync->searchOffset = 0;
	timesync->searchScale = 0;
	timesync->searchInitialSize = timesync->settings->firstSearchInitialSize;
	timesync->searchGrowth = timesync->settings->firstSearchGrowth;


	// Difference in start time between dependent and master samples
	if (timesync->dependentSamples.startTime != 0.0f && timesync->masterSamples.startTime != 0.0f)
	{
		char masterStart[TIME_MAX_STRING];
		char dependentStart[TIME_MAX_STRING];
		TimeString(timesync->masterSamples.startTime, masterStart);
		TimeString(timesync->dependentSamples.startTime, dependentStart);

		timesync->dependentTimeOffset = timesync->dependentSamples.startTime - timesync->masterSamples.startTime;
		timesync->dependentSampleOffset = (int)(timesync->dependentTimeOffset * timesync->sampleRate);

		fprintf(stderr, "NOTE: Start time present in both files, master '%s', dependent '%s', difference: %f (%d samples)\n", masterStart, dependentStart, timesync->dependentTimeOffset, timesync->dependentSampleOffset);

		if ((unsigned int)abs(timesync->dependentSampleOffset) > timesync->masterSamples.numSamples && (unsigned int)abs(timesync->dependentSampleOffset) > timesync->dependentSamples.numSamples)
		{
			fprintf(stderr, "WARNING: Times in both files do not overlap, assuming same start.\n");
			timesync->dependentTimeOffset = 0.0f;
			timesync->dependentSampleOffset = 0;
		}
	}
	else
	{
		timesync->dependentTimeOffset = 0.0f;
		timesync->dependentSampleOffset = 0;
		fprintf(stderr, "WARNING: Start time not present in both files, assuming same start.\n");
	}

	// Thread data
	process_data_t processDataBlock[MAX_PROCESS_COUNT] = { { 0 } };
	thread_t threads[MAX_PROCESS_COUNT] = { 0 };
	timesync->processCount = timesync->settings->processCount;
	if (timesync->processCount <= 0)
	{
		// Fetch number of processors (plus any additional threads)
		timesync->processCount = get_nprocs() + abs(timesync->processCount);
	}
	if (timesync->processCount > MAX_PROCESS_COUNT)
	{
		timesync->processCount = MAX_PROCESS_COUNT;
	}
	fprintf(stderr, "Creating %d threads...\n", timesync->processCount);

	event_t eventFinish;
	event_init(&eventFinish);
	for (int j = 0; j < timesync->processCount; j++)
	{ 
		processDataBlock[j].name = j;
		processDataBlock[j].active = false;
		processDataBlock[j].i = -1;
		event_init(&processDataBlock[j].eventStart);
		thread_create(&threads[j], NULL, (thread_start_routine_t)Process, &processDataBlock[j]);
	}

	// Process
	fprintf(stderr, "Processing...\n");
	int lastPercent = 0;
	double startTime = TimeNow();
	timesync->windowsStarted = 0;
	timesync->windowsComplete = 0;
	while (timesync->windowsComplete < timesync->numWindows)
	{
		// Process threads
		bool anyStopped;
		do
		{
			anyStopped = false;
			for (int j = 0; j < timesync->processCount; j++)
			{
				process_data_t *processData = &processDataBlock[j];
				// If this one is not running...
				if (!processData->active)
				{
					int start = -1;			// assume not starting

					// If not used, start the next one
					if (processData->i < 0)
					{
//fprintf(stderr, "#%d: not used, will start %d\n", j, start);
						start = timesync->windowsStarted++;
					}
					else if (processData->i == timesync->windowsComplete)	// ...if it is the next output we expect
					{
						// If we're discarding this output (the settings changed)
						if (processData->discard)
						{
							// Start this one again
							start = processData->i;
						}
						else
						{
							// Start the next one
							start = timesync->windowsStarted++;
//fprintf(stderr, "#%d: finished %d, will start %d\n", j, processData->i, start);

							// Copy result
							timesync->correlationResults[timesync->windowsComplete] = processData->correlationResult;	// copy
							timesync->windowsComplete++;

#if 0
							// Output time of best coeficient
							float bestCoefficient;
							float bestCorrelationTime = BestCorrelation(timesync, &timesync->correlationResults[timesync->windowsComplete - 1], &bestCoefficient);
							double masterOffsetTime = processData->masterOffsetTime;
							printf("%d,%.3f,", timesync->windowsComplete, masterOffsetTime);
							if (bestCorrelationTime != INFINITY)
							{
								printf("%.3f,%.3f\n", bestCorrelationTime, bestCoefficient);
							}
							else
							{
								printf(",\n");
							}
#endif

							// Periodically analyse the output
							if (TimeSyncAnalyse(timesync))
							{
								// Settings changed, invalidate all the ones in progress
								for (int z = 0; z < timesync->processCount; z++)
								{
									processDataBlock[z].discard = true;
								}
							}

							// Report progress
							int percent = 100 * timesync->windowsComplete / timesync->numWindows;
							if (lastPercent != percent)
							{
								double now = TimeNow();
								double elapsed = now - startTime;
								double remaining = (timesync->numWindows - timesync->windowsComplete) * elapsed / timesync->windowsComplete;
								lastPercent = percent;
								fprintf(stderr, "\r@ %2d%% in %d:%02d, %d:%02d remaining...", percent, (int)elapsed / 60, (int)elapsed % 60, (int)remaining / 60, (int)remaining % 60);
								fflush(stdout);
							}
						}

						// Flag as invalid
						processData->i = -1;
						anyStopped = true;
					}

					// If we're starting a new one...
					if (start >= 0 && start < timesync->numWindows)
					{
						double masterOffsetTime = start * timesync->settings->windowSkipTime;

//fprintf(stderr, "#%d: start %d\n", j, start);

						process_data_t *processData = &processDataBlock[j];
						processData->i = start;
						processData->timesync = timesync;
						processData->masterOffsetTime = masterOffsetTime;

						double centre = (timesync->searchOffset + timesync->searchScale * processData->i) / timesync->sampleRate;
						double size = timesync->searchInitialSize + ((double)(processData->i - timesync->searchLastWindow) * timesync->settings->windowSizeTime) * timesync->searchGrowth;

						processData->minShiftTime = centre - (size / 2);
						processData->maxShiftTime = centre + (size / 2);

						processData->windowSizeTime = timesync->settings->windowSizeTime;
						processData->eventFinish = &eventFinish;
						processData->active = true;
						processData->discard = false;

//fprintf(stderr, "M:signal-start-%d @%d\n", processData->name, processData->i);
						event_signal(&processData->eventStart);
					}
				}
			}


		} while (anyStopped);
//fprintf(stderr, "---\n");

		if (timesync->windowsComplete >= timesync->numWindows)
		{
			break;
		}

//fprintf(stderr, "M:wait-finish\n");
		event_wait(&eventFinish);
//fprintf(stderr, "M:~wait-finish\n");
	}

	// Kill threads
	fprintf(stderr, "Stopping threads...\n");
	for (int j = 0; j < timesync->processCount; j++)
	{
		//fprintf(stderr, "Signalling thread %d...\n", j);
		processDataBlock[j].quit = true;
		event_signal(&processDataBlock[j].eventStart);
		//fprintf(stderr, "Terminating thread %d...\n", j);
		//thread_cancel(threads[j]);
		//fprintf(stderr, "Waiting for thread %d...\n", j);
		thread_join(threads[j], NULL);
		//fprintf(stderr, "Destroying event %d...\n", j);
		event_destroy(&processDataBlock[j].eventStart);
	}
	event_destroy(&eventFinish);

	// Overall best fit
	TimeSyncCalcOverallBestFit(timesync);

	// Output debug image
	TimeSyncOutputImage(timesync);

	// Output resampled file
	TimeSyncOutputResampled(timesync);

	// Output CSV file
	TimeSyncOutputCsv(timesync);

	// Free results
	fprintf(stderr, "Freeing...\n");
	for (int i = 0; i < timesync->numWindows; i++)
	{
		if (timesync->correlationResults[i].correlationCoefficients != NULL)
		{
			free(timesync->correlationResults[i].correlationCoefficients);
		}
	}
	free(timesync->correlationResults);
	free(timesync->bestFitShift);


	char *result;
	int retVal;
	if (timesync->numValid == 0)
	{
		result = "None";
		retVal = 2;
	}
	else if (timesync->numValid < timesync->numRates)
	{
		result = "Some";
		retVal = 1;
	}
	else
	{
		result = "All";
		retVal = 0;
	}


	fprintf(stderr, "Done! Result: %d=%s\n", retVal, result);

	return retVal;
}


// Run the time synchronization algorithm using the provided settings
int TimeSyncRun(timesync_settings_t *settings)
{
	timesync_t timesyncState = { 0 };
	timesync_t *timesync = &timesyncState;
	int ret;

	// Open the time sync
	ret = TimeSyncOpen(timesync, settings);
	if (ret != EXIT_OK)	{ return ret; }

	// Process the time sync
	ret = TimeSyncProcess(timesync);
	
	// Close the time sync
	TimeSyncClose(timesync);

	return ret;
}

