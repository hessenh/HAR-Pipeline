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

#ifndef TIMESYNC_H
#define TIMESYNC_H

// Time synchronization settings
typedef struct
{
	const char *masterFilename;
	const char *dependentFilename;
	const char *outFilename;
	const char *csvFilename;
	const char *imageFilename;

	double csvStartTime;
	double csvEndTime;
	int csvSkip;

	int processCount;							// Number of threads (<=0 auto plus abs)

	double windowSizeTime;						// Size of each window (seconds)
	double windowSkipTime;						// Skip amount between each window (not tested for anything other than windowSizeTime)

	double analysisSkipTime;					// Time between each analysis

	double firstSearchInitialSize;				// Initial size for the first search
	double firstSearchGrowth;					// Growth rate for the first search

	double searchInitialSize;					// Initial size for a continuing search
	double searchGrowth;						// Growth rate for a continuing search

	int maxIterations;							// RANSAC iteration count

	double medianWindowTime;					// Window for median filter
	double minStdDev;							// Minimum std-dev in window
	double medianMaxTime;						// Maximum distance from median value

	double analysisMaxTime;						// Maximum distance from the line
	double analysisMinimumPointsProportion;		// Minimum number of points in an analysis
	double analysisMinimumRangeProportion;		// Minimum proportion of points in an analysis within the maximum range

} timesync_settings_t;

// Initialize the settings to default values
void TimeSyncDefaults(timesync_settings_t *settings);

// Run the time synchronization algorithm using the provided settings
int TimeSyncRun(timesync_settings_t *settings);

#endif

