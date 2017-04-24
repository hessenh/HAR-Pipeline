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

// .BMP Bitmap file header writer
// Dan Jackson

#ifndef BITMAP_H
#define BITMAP_H


// Notes
// * Use a negative height to indicate a top-down bitmap (default is bottom-up)
// * Use 32 bpp for BGRX (?)


// Usage:
//		FILE *bmpFp = fopen(bmpFile, "w+b");
//		fseek(bmpFp, 0, SEEK_SET);	// Can reset to finalize a bitmap with previously unknown height
//		fwrite(BitmapWriteHeader(NULL, NULL, width, height, 32), 1, BMP_WRITER_SIZE_HEADER, bmpFp);


// Length of a bitmap header
#define BMP_WRITER_SIZE_HEADER 54

// Debug BMP Header writing
static void *BitmapHeader(void *buffer, int *size, int width, int height, int bitsPerPixel)
{
	static unsigned char staticBuffer[BMP_WRITER_SIZE_HEADER];
	const unsigned int headerSize = BMP_WRITER_SIZE_HEADER;	// 54;											// Header size (54)
	const unsigned int paletteSize = ((bitsPerPixel <= 8) ? ((unsigned int)1 << bitsPerPixel) : 0) << 2;	// Number of palette bytes
	const unsigned int stride = 4 * ((width * ((bitsPerPixel + 7) / 8) + 3) / 4);							// Byte width of each line
	const unsigned long biSizeImage = (unsigned long)stride * (height < 0 ? -height : height);				// Total number of bytes that will be written
	const unsigned long bfOffBits = headerSize + paletteSize;
	const unsigned long bfSize = bfOffBits + biSizeImage;
	unsigned char *p = (unsigned char *)buffer;
	if (p == 0) { p = staticBuffer; }
	memset(buffer, 0, headerSize);	// Unset bytes are zero
	p[0] = 'B'; p[1] = 'M';																																					// @0 WORD bfType
	p[2] = (unsigned char)bfSize; p[3] = (unsigned char)(bfSize >> 8); p[4] = (unsigned char)(bfSize >> 16); p[5] = (unsigned char)(bfSize >> 24);							// @2 DWORD bfSize, @6 WORD bfReserved1, @8 WORD bfReserved2
	p[10] = (unsigned char)bfOffBits; p[11] = (unsigned char)(bfOffBits >> 8); p[12] = (unsigned char)(bfOffBits >> 16); p[13] = (unsigned char)(bfOffBits >> 24);			// @10 DWORD bfOffBits
	p[14] = 40;																																								// @14 DWORD biSize
	p[18] = (unsigned char)width; p[19] = (unsigned char)(width >> 8); p[20] = (unsigned char)(width >> 16); p[21] = (unsigned char)(width >> 24);							// @18 DWORD biWidth
	p[22] = (unsigned char)height; p[23] = (unsigned char)(height >> 8); p[24] = (unsigned char)(height >> 16); p[25] = (unsigned char)(height >> 24);						// @22 DWORD biHeight
	p[26] = 1;																																								// @26 WORD biPlanes
	p[28] = bitsPerPixel;																																					// @28 WORD biBitCount, @30 DWORD biCompression (0=BI_RGB, 3=BI_BITFIELDS)
	p[34] = (unsigned char)biSizeImage; p[35] = (unsigned char)(biSizeImage >> 8); p[36] = (unsigned char)(biSizeImage >> 16); p[37] = (unsigned char)(biSizeImage >> 24);	// @34 biSizeImage, @38 DWORD biXPelsPerMeter, @42 DWORD biYPelsPerMeter, @46 DWORD biClrUsed, @50 DWORD biClrImportant, @54 <end>
	if (size != 0) { *size = headerSize; }
	return p;
}

#endif
