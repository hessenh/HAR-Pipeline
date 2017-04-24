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

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#define strcasecmp _stricmp
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "exits.h"
#include "timestamp.h"
#include "timesync.h"


int main(int argc, char *argv[])
{
	int i;
	char help = 0;
	int positional = 0;
	int ret;
	timesync_settings_t settings = { 0 };

#ifdef _WIN32
//	static char stdoutbuf[20];
//	static char stderrbuf[20];
//	setvbuf(stdout, stdoutbuf, _IOFBF, sizeof(2));
//	setvbuf(stderr, stderrbuf, _IOFBF, sizeof(2));
#endif

	// Default settings
	TimeSyncDefaults(&settings);

	for (i = 1; i < argc; i++)
	{
		int remaining = argc - 1 - i;
		if (strcmp(argv[i], "--help") == 0) { help = 1; }
		else if (strcmp(argv[i], "-out") == 0 && remaining >= 1) { settings.outFilename = argv[++i]; }
		else if (strcmp(argv[i], "-bmp") == 0 && remaining >= 1) { settings.imageFilename = argv[++i]; }
		else if (strcmp(argv[i], "-csv") == 0 && remaining >= 1) { settings.csvFilename = argv[++i]; }
		else if (strcmp(argv[i], "-csvstart") == 0 && remaining >= 1) { settings.csvStartTime = TimeParse(argv[++i]); }
		else if (strcmp(argv[i], "-csvend") == 0 && remaining >= 1) { settings.csvEndTime = TimeParse(argv[++i]); }
		else if (strcmp(argv[i], "-csvskip") == 0 && remaining >= 1) { settings.csvSkip = atoi(argv[++i]); }
		else if (argv[i][0] == '-')
		{
			fprintf(stderr, "Unknown option, or option missing required parameters: %s\n", argv[i]);
			help = 1;
		}
		else
		{
			if (positional == 0)
			{
				settings.masterFilename = argv[i];
			}
			else if (positional == 1)
			{
				settings.dependentFilename = argv[i];
			}
			else
			{
				fprintf(stderr, "Unknown positional parameter (%d): %s\n", positional + 1, argv[i]);
				help = 1;
			}
			positional++;
		}
	}


	if (settings.masterFilename == NULL) { fprintf(stderr, "ERROR: Master input file not specified.\n"); help = 1; }
	if (settings.dependentFilename == NULL) { fprintf(stderr, "ERROR: Dependent input file not specified.\n"); help = 1; }
	//if (settings.outFilename == NULL && settings.svmFilename == NULL) { fprintf(stderr, "ERROR: Output/SVM file not specified.\n"); help = 1; }

	if (help)
	{
		fprintf(stderr, "Usage: timesync <master.wav> <dependent.wav> [<options>...]\n");
		fprintf(stderr, "\n");
		fprintf(stderr, "Where <options> are:\n");
		fprintf(stderr, "\n");
		fprintf(stderr, "\t-out <output.wav>                       Resampling of the dependent file synchronized to the master\n");
		fprintf(stderr, "\t-csv <output.csv>                       CSV output of the master and dependent file\n");
		fprintf(stderr, "\t-csvstart <\"YYYY-MM-DD hh:mm:ss.fff\">   Specify CSV output start date/time\n");
		fprintf(stderr, "\t-csvend <\"YYYY-MM-DD hh:mm:ss.fff\">     Specify CSV output end date/time\n");
		fprintf(stderr, "\t-csvskip <num.samples>                  CSV output sample pitch (default 1)\n");
		fprintf(stderr, "\t-bmp <image.bmp>                        Output debug image plot of the correlation\n");
		fprintf(stderr, "\n");

		ret = EXIT_USAGE;
	}
	else
	{
		// Run converter
		ret = TimeSyncRun(&settings);
	}

#if defined(_WIN32) && defined(_DEBUG)
	if (IsDebuggerPresent()) { fprintf(stderr, "\nPress [enter] to exit <%d>....", ret); getc(stdin); }
#endif

	return ret;
}

