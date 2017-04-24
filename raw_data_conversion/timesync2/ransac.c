/*
* Copyright (c) 2014-2017, Newcastle University, UK.
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

// RANSAC (RANdom SAmple Consensus) for Linear Regression
// Dan Jackson, 2014-2017

// Algorithm from "Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography." Communications of the ACM, 24(6) : 381�395, 1981.

#include <alloca.h>
#define _alloca alloca

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "ransac.h"
#include "linearregression.h"

//#define DEBUG



// Small pseudo-random number generator by Bob Jenkins from http://burtleburtle.net/bob/rand/smallprng.html
typedef struct { uint32_t a; uint32_t b; uint32_t c; uint32_t d; } rnd_state;

#define rot(x, k) ((uint32_t)(((x)<<(k))|((x)>>(32-(k)))))

static uint32_t rnd_val(rnd_state *x) 
{
	uint32_t e = x->a - rot(x->b, 27);
	x->a = x->b ^ rot(x->c, 17);
	x->b = x->c + x->d;
	x->c = x->d + e;
	x->d = e + x->a;
	return x->d;
}

static void rnd_init(rnd_state *x, uint32_t seed) 
{
	uint32_t i;
	x->a = 0xf1ea5eed, x->b = x->c = x->d = seed;
	for (i = 0; i < 20; ++i) 
	{
		(void)rnd_val(x);
	}
}



// RANSAC for linear regression with one independent variable
bool RansacLinearModelFit(double *coef, int n, double *y, double *x, int minimumPoints, double tolerance, int maxIterations, double minimumFitFraction, int *outputFlags)
{
	// Dummy/default output parameters
	coef[0] = 0.0f;		// offset (intersect)
	coef[1] = 0.0f;		// scale (gradient)

	// Must have at least the minimum number of points
	if (n < minimumPoints)
	{
		return false;
	}

	// Flags which points are included
	#define FLAG_INCLUDED   0x01
	#define FLAG_INLIER     0x02
	int *flags = (int *)_alloca(sizeof(int) * n);

	// Scratch x/y values for included points
	double *xx = (double *)_alloca(sizeof(double) * n);
	double *yy = (double *)_alloca(sizeof(double) * n);

	// Best fit
	double bestFitFraction = 0;	// best fit fraction within tolerance
	double bestFit[2] = { 0 };	// best offset (intersect) & scale (gradient)

	// Deterministic results
	rnd_state rndState = { 0 };
	rnd_init(&rndState, 0);
	
	// Loop
	for (int iterations = 0; iterations < maxIterations; iterations++)
	{
#ifdef DEBUG
		printf("---\n");
		printf("ITERATION: %d\n", iterations);
#endif

		// Step 1. Select randomly the minimum number of points required to determine the model parameters.
		memset(flags, 0, sizeof(int) * n);	// clear all flags
		int numPoints = 0;
		while (true)
		{
			int i = rnd_val(&rndState) % n; // n * rand() / RAND_MAX;
			if ((flags[i] & FLAG_INCLUDED) == 0)
			{
				flags[i] |= FLAG_INCLUDED;
				numPoints++;
#ifdef DEBUG
				printf("Point %d/%d: %d\n", numPoints, minimumFitFraction, i);
#endif
				if (numPoints >= minimumPoints) { break; }
			}
		}


		// Step 2. Solve for the parameters of the model.
		numPoints = 0;
		for (int i = 0; i < n; i++)
		{
			if (flags[i] & FLAG_INCLUDED)
			{
				xx[numPoints] = x[i];
				yy[numPoints] = y[i];
				++numPoints;
			}
		}
		double fit[2];
		LinearModelFitOneIndependent(fit, numPoints, yy, xx);
#ifdef DEBUG
		printf("Regression: offset=%f scale=%f\n", fit[0], fit[1]);
#endif


		// Step 3. Determine how many points from the set of all points fit with a predefined tolerance.
		int numFits = 0;
		for (int i = 0; i < n; i++)
		{
			double testY = fit[1] * x[i] + fit[0];
			double diff = fabs(y[i] - testY);
			if (diff <= tolerance)
			{
				flags[i] |= FLAG_INLIER;
				numFits++;
			}
		}

		// Step 4. If the fraction of the number of inliers over the total number points in the set exceeds a predefined threshold re-estimate the model parameters using all the identified inliers and terminate.
		double fitFraction = (double)numFits / n;
#ifdef DEBUG
		printf("Fit-fraction: %d / %d = %f\n", numFits, n, fitFraction);
#endif
		if (fitFraction > bestFitFraction)
		{
#ifdef DEBUG
			printf("Best (better than %f)\n", bestFitFraction);
#endif
			bestFitFraction = fitFraction;

			// (re-estimate the model parameters using all the identified inliers)
			numPoints = 0;
			for (int i = 0; i < n; i++)
			{
				if (flags[i] & FLAG_INLIER)
				{
					xx[numPoints] = x[i];
					yy[numPoints] = y[i];
					++numPoints;
				}
			}
			LinearModelFitOneIndependent(bestFit, numPoints, yy, xx);
#ifdef DEBUG
			printf("Better-regression: numPoints=%d offset=%f scale=%f\n", numPoints, bestFit[0], bestFit[1]);
#endif

			// Output flags
			if (outputFlags != NULL)
			{
				memcpy(outputFlags, flags, sizeof(int) * n);
			}
		}

		// Step 5. Otherwise, repeat steps 1 through 4 (maximum of N times).
		// (repeat for best fit anyway)
	}
	
	bool successful = (bestFitFraction > minimumFitFraction);
#ifdef DEBUG
	printf("Successful=%s\n", successful ? "yes" : "no");
	printf("---\n");
#endif
	coef[0] = bestFit[0];		// offset (intersect)
	coef[1] = bestFit[1];		// scale (gradient)

	return successful;
}
