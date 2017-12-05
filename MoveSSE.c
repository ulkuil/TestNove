#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "xmmintrin.h"


#define stdcall __attribute__((regparm(3)))


// --------------------------------------------------------

#define LoadCharsSSE(Count, s) \
	switch(Count) { \
		case 8: xmm8 = _mm_load_si128((__m128i*)(s) + 7); \
		case 7: xmm7 = _mm_load_si128((__m128i*)(s) + 6); \
		case 6: xmm6 = _mm_load_si128((__m128i*)(s) + 5); \
		case 5: xmm5 = _mm_load_si128((__m128i*)(s) + 4); \
		case 4: xmm4 = _mm_load_si128((__m128i*)(s) + 3); \
		case 3: xmm3 = _mm_load_si128((__m128i*)(s) + 2); \
		case 2: xmm2 = _mm_load_si128((__m128i*)(s) + 1); \
		case 1: xmm1 = _mm_load_si128((__m128i*)(s)); \
		case 0: break; \
	}
	
#define GetTypev(Getv, Type, v, Offset) \
	Getv = *((Type*)(v + Offset))
	
	
#define LoadCaseSSE(i, r, s) \
	case i+1: \
		{ LoadCharsSSE(r, s); } \
		GetTypev(MByte, uint8_t, s, 16 * r); \
	break; \
	case i+2: \
		{ LoadCharsSSE(r, s); } \
		GetTypev(MWord, uint16_t, s, 16 * r); \
	break; \
	case i+3: \
		{ LoadCharsSSE(r, s); } \
		GetTypev(MWord, uint16_t, s, (16 * r)); \
		GetTypev(MByte, uint8_t, s, (16 * r) + 2); \
	break; \
	case i+4: \
		{ LoadCharsSSE(r, s); } \
		GetTypev(MDowrd, uint32_t, s, 16 * r); \
	break; \
	case i+5: \
		{ LoadCharsSSE(r, s); } \
		GetTypev(MDowrd, uint32_t, s, (16 * r)); \
		GetTypev(MByte, uint8_t, s, (16 * r) + 4); \
	break; \
	case i+6: \
		{ LoadCharsSSE(r, s); } \
		GetTypev(MDowrd, uint32_t, s, (16 * r)); \
		GetTypev(MWord, uint16_t, s, (16 * r) + 4); \
	break; \
	case i+7: \
		{ LoadCharsSSE(r, s); } \
		GetTypev(MDowrd, uint32_t, s, (16 * r)); \
		GetTypev(MDowrd2, uint32_t, s, (16 * r) + 3); \
	break; \
	case i+8: \
		{ LoadCharsSSE(r, s); } \
		GetTypev(MQword, uint64_t, s, 16 * r); \
	break; \
	case i+9: \
		{ LoadCharsSSE(r, s); } \
		GetTypev(MQword, uint64_t, s, (16 * r)); \
		GetTypev(MByte, uint8_t, s, (16 * r) + 8); \
	break; \
	case i+10: \
		{ LoadCharsSSE(r, s); } \
		GetTypev(MQword, uint64_t, s, (16 * r)); \
		GetTypev(MWord, uint16_t, s, (16 * r) + 8); \
	break; \
	case i+11: \
		{ LoadCharsSSE(r, s); } \
		GetTypev(MQword, uint64_t, s, (16 * r)); \
		GetTypev(MDowrd, uint32_t, s, (16 * r) + 7); \
	break; \
	case i+12: \
		{ LoadCharsSSE(r, s); } \
		GetTypev(MQword, uint64_t, s, (16 * r)); \
		GetTypev(MDowrd, uint32_t, s, (16 * r) + 8); \
	break; \
	case i+13: \
		{ LoadCharsSSE(r, s); } \
		GetTypev(MQword, uint64_t, s, (16 * r)); \
		GetTypev(MDowrd, uint32_t, s, (16 * r) + 8); \
		GetTypev(MByte, uint8_t, s, (16 * r) + 8 + 4); \
	break; \
	case i+14: \
		{ LoadCharsSSE(r, s); } \
		GetTypev(MQword, uint64_t, s, (16 * r)); \
		GetTypev(MDowrd, uint32_t, s, (16 * r) + 8); \
		GetTypev(MWord, uint16_t, s, (16 * r) + 8 + 4); \
	break; \
	case i+15: \
		{ LoadCharsSSE(r, s); } \
		GetTypev(MQword, uint64_t, s, (16 * r)); \
		GetTypev(MQword2, uint64_t, s, (16 * r) + 7); \
	break; \
	case i+16: \
		{ LoadCharsSSE(r+1, s); } \
	break;


	
// --------------------------------------------------------
#define MoveLoad(nn, scr, align) align ? \
	_mm_store_si128(scr, nn) : _mm_storeu_si128(scr, nn);
	
#define SetCharsSSE(Count, d, align) \
	switch(Count) { \
		case 8: MoveLoad(xmm8, (__m128i*)(d) + 7, align); \
		case 7: MoveLoad(xmm7, (__m128i*)(d) + 6, align); \
		case 6: MoveLoad(xmm6, (__m128i*)(d) + 5, align); \
		case 5: MoveLoad(xmm5, (__m128i*)(d) + 4, align); \
		case 4: MoveLoad(xmm4, (__m128i*)(d) + 3, align); \
		case 3: MoveLoad(xmm3, (__m128i*)(d) + 2, align); \
		case 2: MoveLoad(xmm2, (__m128i*)(d) + 1, align); \
		case 1: MoveLoad(xmm1, (__m128i*)(d), align); \
		case 0: break; \
	}
	

#define SetTypev(Box, Type, Value, Offset) \
	*((Type*)(Value + Offset)) = Box;	
	
#define SetCaseSSE(i, r, d, align) \
	case i+1: \
		{ SetCharsSSE(r, d, align); } \
		SetTypev(MByte, uint8_t, d, 16 * r); \
	break; \
	case i+2: \
		{ SetCharsSSE(r, d, align); } \
		SetTypev(MWord, uint16_t, d, 16 * r); \
	break; \
	case i+3: \
		{ SetCharsSSE(r, d, align); } \
		SetTypev(MWord, uint16_t, d, (16 * r)); \
		SetTypev(MByte, uint8_t, d, (16 * r) + 2); \
	break; \
	case i+4: \
		{ SetCharsSSE(r, d, align); } \
		SetTypev(MDowrd, uint32_t, d, 16 * r); \
	break; \
	case i+5: \
		{ SetCharsSSE(r, d, align); } \
		SetTypev(MDowrd, uint32_t, d, (16 * r)); \
		SetTypev(MByte, uint8_t, d, (16 * r) + 4); \
	break; \
	case i+6: \
		{ SetCharsSSE(r, d, align); } \
		SetTypev(MDowrd, uint32_t, d, (16 * r)); \
		SetTypev(MWord, uint16_t, d, (16 * r) + 4); \
	break; \
	case i+7: \
		{ SetCharsSSE(r, d, align); } \
		SetTypev(MDowrd, uint32_t, d, (16 * r)); \
		SetTypev(MDowrd2, uint32_t, d, (16 * r) + 3); \
	break; \
	case i+8: \
		{ SetCharsSSE(r, d, align); } \
		SetTypev(MQword, uint64_t, d, 16 * r); \
	break; \
	case i+9: \
		{ SetCharsSSE(r, d, align); } \
		SetTypev(MQword, uint64_t, d, (16 * r)); \
		SetTypev(MByte, uint8_t, d, (16 * r) + 8); \
	break; \
	case i+10: \
		{ SetCharsSSE(r, d, align); } \
		SetTypev(MQword, uint64_t, d, (16 * r)); \
		SetTypev(MWord, uint16_t, d, (16 * r) + 8); \
	break; \
	case i+11: \
		{ SetCharsSSE(r, d, align); } \
		SetTypev(MQword, uint64_t, d, (16 * r)); \
		SetTypev(MDowrd, uint32_t, d, (16 * r) + 7); \
	break; \
	case i+12: \
		{ SetCharsSSE(r, d, align); } \
		SetTypev(MQword, uint64_t, d, (16 * r)); \
		SetTypev(MDowrd, uint32_t, d, (16 * r) + 8); \
	break; \
	case i+13: \
		{ SetCharsSSE(r, d, align); } \
		SetTypev(MQword, uint64_t, d, (16 * r)); \
		SetTypev(MDowrd, uint32_t, d, (16 * r) + 8); \
		SetTypev(MByte, uint8_t, d, (16 * r) + 8 + 4); \
	break; \
	case i+14: \
		{ SetCharsSSE(r, d, align); } \
		SetTypev(MQword, uint64_t, d, (16 * r)); \
		SetTypev(MDowrd, uint32_t, d, (16 * r) + 8); \
		SetTypev(MWord, uint16_t, d, (16 * r) + 8 + 4); \
	break; \
	case i+15: \
		{ SetCharsSSE(r, d, align); } \
		SetTypev(MQword, uint64_t, d, (16 * r)); \
		SetTypev(MQword2, uint64_t, d, (16 * r) + 7); \
	break; \
	case i+16: \
		{ SetCharsSSE(r+1, d, align); } \
	break;








// ---------------------------------------------------------------------

#define SSELoad(align, S, B) B = align ? \
	_mm_load_si128(S) : _mm_loadu_si128(S);

#define MoveSSECount(Count, d, s, align) \
	__m128i xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;\
	switch(Count) { \
		case 8: SSELoad(align, (__m128i*)(s) + 7, xmm8); \
		case 7: SSELoad(align, (__m128i*)(s) + 6, xmm7); \
		case 6: SSELoad(align, (__m128i*)(s) + 5, xmm6); \
		case 5: SSELoad(align, (__m128i*)(s) + 4, xmm5); \
		case 4: SSELoad(align, (__m128i*)(s) + 3, xmm4); \
		case 3: SSELoad(align, (__m128i*)(s) + 2, xmm3); \
		case 2: SSELoad(align, (__m128i*)(s) + 1, xmm2); \
		case 1: SSELoad(align, (__m128i*)(s), xmm1); \
		case 0: break; \
	} \
	switch(Count) { \
		case 8: MoveLoad(xmm8, (__m128i*)(d) + 7, align); \
		case 7: MoveLoad(xmm7, (__m128i*)(d) + 6, align); \
		case 6: MoveLoad(xmm6, (__m128i*)(d) + 5, align); \
		case 5: MoveLoad(xmm5, (__m128i*)(d) + 4, align); \
		case 4: MoveLoad(xmm4, (__m128i*)(d) + 3, align); \
		case 3: MoveLoad(xmm3, (__m128i*)(d) + 2, align); \
		case 2: MoveLoad(xmm2, (__m128i*)(d) + 1, align); \
		case 1: MoveLoad(xmm1, (__m128i*)(d), align); \
		case 0: break; \
	}

#define SetType(t, s, d, o) *((t*)(s + o)) = *((t*)(d + o))

#define Mov1B(d, s, o) SetType(uint8_t, s, d, o)
#define Mov2B(d, s, o) SetType(uint16_t, s, d, o)
#define Mov4B(d, s, o) SetType(uint32_t, s, d, o)
#define Mov8B(d, s, o) SetType(uint64_t, s, d, o)

#define Mov3B(d, s, o) Mov1B(d, s, o); Mov2B(d, s, o + 1); 
#define Mov5B(d, s, o) Mov4B(d, s, o); Mov1B(d, s, o + 4);
#define Mov6B(d, s, o) Mov4B(d, s, o); Mov2B(d, s, o + 4);
#define Mov7B(d, s, o) Mov4B(d, s, o); Mov4B(d, s, o + 3);
#define Mov9B(d, s, o)  Mov8B(d, s, o); Mov1B(d, s, o +8);
#define Mov10B(d, s, o) Mov8B(d, s, o); Mov2B(d, s, o +8);
#define Mov11B(d, s, o) Mov8B(d, s, o); Mov4B(d, s, o + 7);
#define Mov12B(d, s, o) Mov8B(d, s, o); Mov4B(d, s, o + 8);
#define Mov13B(d, s, o) Mov8B(d, s, o); Mov5B(d, s, o + 8);
#define Mov14B(d, s, o) Mov8B(d, s, o); Mov6B(d, s, o + 8);
#define Mov15B(d, s, o) Mov8B(d, s, o); Mov8B(d, s, o + 7);


// -----------------------------------------------------------------------	
	
	
	
stdcall void MoveMem(const char* Source, char* dest, size_t len) {
	__m128i xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
	uint8_t MByte, MByte1;
	uint16_t MWord, MWord2;
	uint32_t MDowrd, MDowrd2;
	uint64_t MQword, MQword2;
	
	
	uint8_t AlignA = len;
	bool c = false;
AlignSet:
	switch(AlignA) {
		case 0: return;
		case 1: Mov1B(dest, Source, 0); break;
		case 2: Mov2B(dest, Source, 0); break;
		case 3: Mov4B(dest, Source, 0); break;
		case 4: Mov4B(dest, Source, 0); break;
		case 5: Mov5B(dest, Source, 0); break;
		case 6: Mov6B(dest, Source, 0); break;
		case 7: Mov7B(dest, Source, 0); break;
		case 8: Mov8B(dest, Source, 0); break;
		case 9: Mov9B(dest, Source, 0); break;
		case 10: Mov10B(dest, Source, 0); break;
		case 11: Mov11B(dest, Source, 0); break;
		case 12: Mov12B(dest, Source, 0); break;
		case 13: Mov13B(dest, Source, 0); break;
		case 14: Mov14B(dest, Source, 0); break;
		case 15: Mov15B(dest, Source, 0); break;
		case 16: MoveSSECount(1, dest, Source, 0); break;
		default: {
			AlignA  = (uint64_t)Source & 15;
			c = AlignA == (uint8_t)((uint64_t)dest & 15);
			
			if(AlignA) goto AlignSet;
		}
	}

	if(AlignA != len) {
		len -= AlignA;
		dest += AlignA;
		Source += AlignA;

		if(c) {
			while(len >= 128) {
				MoveSSECount(8, dest, Source, 1);
				len -= 128;
				dest += 128;
				Source += 128;
			}
		} else {
			while(len >= 128) {
				MoveSSECount(8, dest, Source, 0);
				len -= 128;
				dest += 128;
				Source += 128;
			}
		}
	
		switch(len){ 
			case 0: return; 
			LoadCaseSSE(16 * 0 + 1, 0, Source); 
			LoadCaseSSE(16 * 1 + 1, 1, Source);
			LoadCaseSSE(16 * 2 + 1, 2, Source);
			LoadCaseSSE(16 * 3 + 1, 3, Source);
			LoadCaseSSE(16 * 4 + 1, 4, Source);
			LoadCaseSSE(16 * 5 + 1, 5, Source);
			LoadCaseSSE(16 * 6 + 1, 6, Source);
			LoadCaseSSE(16 * 7 + 1, 7, Source);
		}
		
		if(c) {
			len += 16 * 7 + 1 + 15;
		}

		switch(len){ 
			SetCaseSSE(16 * 0 + 1, 0, dest, 0); 
			SetCaseSSE(16 * 1 + 1, 1, dest, 0);
			SetCaseSSE(16 * 2 + 1, 2, dest, 0);
			SetCaseSSE(16 * 3 + 1, 3, dest, 0);
			SetCaseSSE(16 * 4 + 1, 4, dest, 0);
			SetCaseSSE(16 * 5 + 1, 5, dest, 0);
			SetCaseSSE(16 * 6 + 1, 6, dest, 0);
			SetCaseSSE(16 * 7 + 1, 7, dest, 0);

			SetCaseSSE(16 * 8 + 1, 0, dest, 1); 
			SetCaseSSE(16 * 9 + 1, 1, dest, 1);
			SetCaseSSE(16 * 10 + 1, 2, dest, 1);
			SetCaseSSE(16 * 11 + 1, 3, dest, 1);
			SetCaseSSE(16 * 12 + 1, 4, dest, 1);
			SetCaseSSE(16 * 13 + 1, 5, dest, 1);
			SetCaseSSE(16 * 14 + 1, 6, dest, 1);
			SetCaseSSE(16 * 15 + 1, 7, dest, 1);
		}
	}
}	
