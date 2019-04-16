/*
* Program     : Signal Generator using Time-Varying Room Impulse Responses
*
* Description :
*
* Author      : E.A.P. Habets (ehabets@dereverberation.org)
*
* Version     : 1.5.20170712
*
* History     : 1.0.20080130 Initial version
*               1.0.20080209 Added myPrintf
*               1.1.20080211 Added progress bar
*               1.2.20080713 Minor improvements
*               1.3.20100915 Now uses RIR Generator version 1.9.20090822
*               1.4.20100920 Now uses RIR Generator version 2.0.20100920
*               1.5.20110914 Bug fixes and added support for
*                            time-varying receiver positions
*               1.5.20170712 Changed delete to delete[]
*
* Special thanks go to Mr. Adham Al-Husseiny Mostafa for his contributions
* to version 1.5.20110914.
*
* Copyright (C) 2008-2011 E.A.P Habets
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#define _USE_MATH_DEFINES

//#include "matrix.h"
//#include "mex.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <sys/types.h>

using namespace std;

#define ROUND(x) ((x)>=0?(long)((x)+0.5):(long)((x)-0.5))
#define NO_MICS 4   // Number of microphones
#define ROOM_DEPTH 9
#define ROOM_WIDTH 9
#define ROOM_HEIGHT 3

struct hpf_t {
   double  W;
   double  R1;
   double  B1;
   double  B2;
   double  A1;
};

////////// Additional functions ///////////
__inline__ void mexErrMsgTxt(const char *s) {}


///////////////////////////////////////////

// Modulus operation
int mod(int a, int b) {
   int ret = a % b;
   if(ret < 0)
       ret+=b;
   return ret;
}

double sinc(double x) {
   if (x == 0)
       return(1.);
   else
       return(sin(x)/x);
}

double sim_microphone(double x, double y, double z, double* angle, char mtype) {
   if (mtype=='b' || mtype=='c' || mtype=='s' || mtype=='h') {
       double strength, vartheta, varphi, alpha;

       // Directivity pattern   alpha
       // ---------------------------
       // Bidirectional         0
       // Hypercardioid         0.25
       // Cardioid              0.5
       // Subcardioid           0.75
       // Omnidirectional       1

       switch(mtype) {
           case 'b':
               alpha = 0;
               break;
           case 'h':
               alpha = 0.25;
               break;
           case 'c':
               alpha = 0.5;
               break;
           case 's':
               alpha = 0.75;
               break;
       };

       vartheta = acos(z/sqrt(pow(x, 2)+pow(y, 2)+pow(z, 2)));
       varphi = atan2(y, x);

       strength = sin(M_PI/2-angle[1]) * sin(vartheta) * cos(angle[0]-varphi) + cos(M_PI/2-angle[1]) * cos(vartheta);
       strength = alpha + (1-alpha) * strength;

       return strength;
   }
   else {
       return 1;
   }
}

// Copy impulse response on row_idx-1 to row_idx
void copy_previous_rir(double* imp, int row_idx, int nsamples) {
   if (row_idx == 0) {
       for (int tmp_pos_idx = 0; tmp_pos_idx < nsamples; tmp_pos_idx++)
           imp[nsamples*tmp_pos_idx] = imp[(nsamples-1) + nsamples*tmp_pos_idx];
   }
   else {
       for (int tmp_pos_idx = 0; tmp_pos_idx < nsamples; tmp_pos_idx++)
           imp[row_idx + nsamples*tmp_pos_idx] = imp[(row_idx-1) + nsamples*tmp_pos_idx];
   }
}

// High-pass filter the impulse response
void hpf_imp(double* imp, int row_idx, int nsamples, hpf_t hpf) {
   double* Y = new double[3];
   double X0;

   for (int idx = 0; idx < 3; idx++) {
       Y[idx] = 0;
   }
   for (int idx = 0; idx < nsamples; idx++) {
       X0 = imp[row_idx + nsamples*idx];
       Y[2] = Y[1];
       Y[1] = Y[0];
       Y[0] = hpf.B1*Y[1] + hpf.B2*Y[2] + X0;
       imp[row_idx + nsamples*idx] = Y[0] + hpf.A1*Y[1] + hpf.R1*Y[2];
   }

   delete[] Y;
}

//void mexFunction(int nlhs, void *plhs[], int nrhs, const void *prhs[]) {
// "function [out, beta_hat] = signal_generator(in, c, fs, r_path, s_path, L, beta,\n"
//                 " nsamples, mtype, order, dim, orientation, hp_filter);\n\n"
//                 "Input parameters:\n"
//                 " signal_length: The length of the input signal 'in'.\n"
//                 " in           : Input audio signal.\n"
//                 " fs           : Sampling frequency in Hz.\n"
//                 " r_path       : M x 3 array specifying the (x,y,z) coordinates of the receiver(s) in m.\n"
//                 " s_path       : N x 3 vector specifying the (x,y,z) coordinates of the source\n"
//                 "               in meter for each sample time.\n"
//                 " L           : 1 x 3 vector specifying the room dimensions (x,y,z) in m.\n"
//                 " beta        : 1 x 6 vector specifying the reflection coefficients\n"
//                 "               [beta_x1 beta_x2 beta_y1 beta_y2 beta_z1 beta_z2] or\n"
//                 "               beta = Reverberation Time (T_60) in seconds.\n"
//                 " nsamples    : number of samples to calculate, default is T_60*fs.\n"
//                 " mtype       : 'o' = omnidirectional, 's' = subcardioid, 'c' = cardioid,\n"
//                 "               'h' = hypercardioid, 'b' = bidirectional], default is omnidirectional.\n"
//                 "               Individual directivity patterns can specified using a\n"
//                 "               string of length M.\n"
//                 " order       : reflection order, default is -1, i.e. maximum order.\n"
//                 " dim         : room dimension (2 or 3), default is 3.\n"
//                 " orientation : direction in which the microphones are pointed, specified using\n"
//                 "               azimuth and elevation angles (in radians), default is [0 0].\n"
//                 "               The following options are accepted:\n"
//                 "               scalar: [30]           -> azimuth=30, elevation=0 for all microphones\n"
//                 "               column vector: [30;40] -> number of rows must be equal to M\n"
//                 "                                         (azimuth_1 = 30, elevation_1 = 0,\n"
//                 "                                         azimuth_2 = 40, elevation_2 = 0)\n"
//                 "               row vector: [30,60]    -> azimuth=30, elevation=60 for all microphones\n"
//                 "               matrix: [30,60;40,10]  -> number of rows must be equal to M.\n"
//                 " hp_filter   : use 'false' to disable high-pass filter, the high-pass filter\n"
//                 "               is enabled by default.\n\n"
//                 "Output parameters:\n"
//                 " out         : M x N matrix containing the generated signals.\n"
//                 " beta_hat    : In case a reverberation time is specified as an input parameter\n"
//                 "               the corresponding reflection coefficient is returned.\n\n");
void mexFunction(const long signal_length, const double *in, const double fs, const double *rr, const double *ss, double *out) {
   // Set parameters
   double          c = 340.0;
   double          LL[3] = {10, 10, 3};
   const double *  orientation = NULL;
   //const int       dim = 3;
   const int       hp_filter = 1;          // High-pass filter (optional)
   double          T_60 = 0.2;
   double *        beta = new double[6];
   const char *    mtype_ptr = "o";
   int             nsamples = 1024;
   int             order = 2;

   // Load parameters
   // Number of microphones
   int no_mics = NO_MICS;

   double *        angle = new double[2];
   double *        angles = new double[2*no_mics];
   double          TR;

   // Reflection coefficients or reverberation time?
   if (T_60 != 0.0) {
       double V = LL[0]*LL[1]*LL[2];
       double S = 2*(LL[0]*LL[2]+LL[1]*LL[2]+LL[0]*LL[1]);
       TR = T_60;
       double alfa = 24*V*log(10.0)/(c*S*TR);
       if (alfa > 1)
           mexErrMsgTxt("Error: The reflection coefficients cannot be calculated using the current "
                        "room parameters, i.e. room size and reverberation time.\n           Please "
                        "specify the reflection coefficients or change the room parameters.");
       double temp = sqrt(1-alfa);
       for (int i = 0; i < 6; i++)
           beta[i] = temp;
   }

   // 3D microphone orientation (optional)
   if (orientation != NULL)
   {
       assert(0);
       // const double* orientation = mxGetPr(prhs[11]);

//        if(mxGetN(prhs[11]) == 1 && mxGetM(prhs[11]) == 1)
//        {
//            for (int idx = 0; idx < no_mics; idx++)
//            {
//                angles[idx] = orientation[0];
//                angles[idx+no_mics] = 0;
//            }
//        }
//        else if (mxGetN(prhs[11]) == 2 && mxGetM(prhs[11]) == 1)
//        {
//            for (int idx = 0; idx < no_mics; idx++)
//            {
//                angles[idx] = orientation[0];
//                angles[idx+no_mics] = orientation[1];
//            }
//        }
//        else if (mxGetN(prhs[11]) == 1 && mxGetM(prhs[11]) == no_mics)
//        {
//            for (int idx = 0; idx < no_mics; idx++)
//            {
//                angles[idx] = orientation[idx];
//                angles[idx+no_mics] = 0;
//            }
//        }
//        else if(!(mxGetM(prhs[11]) == 1) && mxGetN(prhs[11]) == 2)
//        {
//            if (!(mxGetM(prhs[11])==no_mics))
//                mexErrMsgTxt("Invalid input argument orientation!");
//            else
//            {
//                for (int idx = 0; idx < 2*no_mics; idx++)
//                    angles[idx] = orientation[idx];
//            }
//        }
//        else
//            mexErrMsgTxt("Invalid input argument orientation!");
   }
   else {
       for (int idx=0; idx < 2*no_mics; idx++)
           angles[idx]=0;
   }

   // Type of microphone (optional)
   char *mtype = new char[no_mics];
   if (mtype_ptr != NULL) {
       if (str_len(mtype_ptr) == 1) {
           for (int mic_idx = 0; mic_idx < no_mics; mic_idx++)
               mtype[mic_idx] = mtype_ptr[0];
       }
       else if (str_len(mtype_ptr) == no_mics) {
           for (int mic_idx = 0; mic_idx < no_mics; mic_idx++)
               mtype[mic_idx] = mtype_ptr[mic_idx];
       }
       else
           mexErrMsgTxt("Invalid input argument mtype!");
   }
   else {
       for (int mic_idx = 0; mic_idx < no_mics; mic_idx++)
           mtype[mic_idx] = 'o';
   }

   // Number of samples (optional)
   if (nsamples == -1) {
       if (T_60 == 0) {
           double V = LL[0] * LL[1] * LL[2];
           //double S = 2 * (LL[0] * LL[2] + LL[1] * LL[2] + LL[0] * LL[1]);
           double alpha = ((1 - pow(beta[0], 2)) + (1 - pow(beta[1], 2))) * LL[0] * LL[2]
                          + ((1 - pow(beta[2], 2)) + (1 - pow(beta[3], 2))) * LL[1] * LL[2]
                          + ((1 - pow(beta[4], 2)) + (1 - pow(beta[5], 2))) * LL[0] * LL[1];
           TR = 24 * log(10.0) * V / (c * alpha);
           if (TR < 0.128)
               TR = 0.128;
       }
       nsamples = (int)(TR * fs);
   }

   // Define high-pass filter
   struct hpf_t hpf;
   hpf.W = 2*M_PI*100/fs;
   hpf.R1 = exp(-hpf.W);
   hpf.B1 = 2*hpf.R1*cos(hpf.W);
   hpf.B2 = -hpf.R1 * hpf.R1;
   hpf.A1 = -(1+hpf.R1);

   // Declarations for image source method
   //mxArray*   imp_mtx = new mxArray(nsamples, nsamples);
   double*      imp = new double[nsamples * nsamples];
   const double cTs = c/fs;
   const int    Tw = 2 * ROUND(0.004*fs);
   double*      LPI = new double[Tw+1];
   printf("LPI = %p, sizeof(double) = %d\n", LPI, sizeof(double));
   double*      hanning_window = new double[Tw+1];
   double*      r = new double[3];
   double*      s = new double[3];
   double*      L = new double[3];
   int*         n = new int[3];

   // Initialization
   for (int idx = 0; idx < 3; idx++)
       L[idx] = LL[idx]/cTs;

   for (int idx = 0; idx < 3; idx++)
       n[idx] = (int) ceil(nsamples/(2*L[idx]));

   for (int idx = 0; idx < Tw+1; idx++)
       hanning_window[idx] = 0.5 * (1 + cos(2*M_PI*(idx+Tw/2)/Tw)); // Hanning window
// printf("22222\n");
   // Process each receiver seperately
   for (int mic_idx = 0; mic_idx < no_mics; mic_idx++) {
       angle[0] = angles[mic_idx];
       angle[1] = angles[mic_idx + no_mics];
//printf("4444\n");
       // Clear response matrix
       for (long counter = 0; counter < nsamples*nsamples; counter++){
           printf("counter = %ld, nsamples = %d, imp = %p\n", counter, nsamples, imp);
           imp[counter] = 0;
       }
//printf("55555\n");
       for (long sample_idx = 0; sample_idx < signal_length; sample_idx++) {
           //char command_string[20];
           int  row_idx_1;
           int  row_idx_2;
           int  no_rows_to_update;
           bool bRcvInvariant_1;
           bool bSrcInvariant_1;
           bool bSrcInvariant_2;
//printf("6666\n");
           // Determine row_idx_1;
           row_idx_1 = sample_idx % nsamples;

           for(int idx=0; idx<3; idx++)
               r[idx] = rr[sample_idx + idx*1 + 3*mic_idx*1]/cTs;
//printf("7777\n");
           if (sample_idx > 0) {
               //bSrcInvariant_1 = IsSrcPosConst(ss, signal_length, sample_idx, 0);
               //bRcvInvariant_1 = IsRcvPosConst(rr, signal_length, sample_idx, mic_idx);
               bSrcInvariant_1 = true;
               bRcvInvariant_1 = true;
           }
           else {
               bSrcInvariant_1 = false;
               bRcvInvariant_1 = false;
           }
//printf("33333\n");
           if ((bRcvInvariant_1 && bSrcInvariant_1) == false) {
               if (bRcvInvariant_1 == false && sample_idx > 0) {
                   if (sample_idx < nsamples)
                       no_rows_to_update = sample_idx;
                   else
                       no_rows_to_update = nsamples;
               }
               else {
                   no_rows_to_update = 1;
               }

               // Update response matrix
               for (int row_counter = 0; row_counter < no_rows_to_update; row_counter++) {

                   row_idx_2 = mod(row_idx_1-row_counter, nsamples);

                   if (row_counter > 0)
                       bSrcInvariant_2 = true;
                       //bSrcInvariant_2 = IsSrcPosConst(ss, signal_length, sample_idx, row_counter);
                   else
                       bSrcInvariant_2 = false;

                   if (bSrcInvariant_2 == false) {
                       double hu[6];
                       double refl[3];
                       int    q, j, k;
                       int    mx, my, mz;

                       // Get source position
                       for(int idx=0;idx<3;idx++)
                           s[idx] = ss[sample_idx - row_counter + idx*1]/cTs;

                       // Clear old impulse response
                       for (int idx = 0; idx < nsamples; idx++)
                           imp[row_idx_2 + nsamples*idx] = 0;

                       // Compute new impulse response
                       for (mx = -n[0]; mx <= n[0]; mx++) {
                           hu[0] = 2*mx*L[0];

                           for (my = -n[1]; my <= n[1]; my++) {
                               hu[1] = 2*my*L[1];

                               for (mz = -n[2]; mz <= n[2]; mz++) {
                                   hu[2] = 2*mz*L[2];

                                   for (q = 0; q <= 1; q++) {
                                       hu[3] = (1-2*q)*s[0] - r[0] + hu[0];
                                       refl[0] = pow(beta[0], abs(mx-q)) * pow(beta[1], abs(mx));

                                       for (j = 0; j <= 1; j++) {
                                           hu[4] = (1-2*j)*s[1] - r[1] + hu[1];
                                           refl[1] = pow(beta[2], abs(my-j)) * pow(beta[3], abs(my));

                                           for (k = 0; k <= 1; k++) {
                                               hu[5] = (1-2*k)*s[2] - r[2] + hu[2];
                                               refl[2] = pow(beta[4], abs(mz-k)) * pow(beta[5], abs(mz));

                                               if (abs(2*mx-q)+abs(2*my-j)+abs(2*mz-k) <= order || order == -1) {
                                                   double dist = sqrt(pow(hu[3], 2) + pow(hu[4], 2) + pow(hu[5], 2));
                                                   int fdist = (int) floor(dist);
                                                   if (fdist < nsamples) {
                                                       for (int idx = 0; idx < Tw+1; idx++){
                                                           const double Fc = 1;
                                                           LPI[idx] = hanning_window[idx] * Fc * sinc(M_PI*Fc*(idx-(dist-fdist)-(Tw/2)));
                                                       }

                                                       for (int idx = 0; idx < Tw+1; idx++) {
                                                           int pos = fdist-(Tw/2);
                                                           if (pos+idx >= 0 && pos+idx < nsamples) {
                                                               double strength = sim_microphone(hu[3], hu[4], hu[5], angle, mtype[mic_idx])
                                                                                 * refl[0]*refl[1]*refl[2]/(4*M_PI*dist*cTs);
                                                               imp[row_idx_2 + nsamples*(pos+idx)] += strength * LPI[idx];
                                                           }
                                                       }
                                                   }
                                               }
                                           }
                                       }
                                   }
                               }
                           }
                       }

                       // Apply original high-pass filter as proposed by Allen and Berkley
                       if (hp_filter == 1) {
                           hpf_imp(imp, row_idx_2, nsamples, hpf);
                       }
                   }
                   else {
                       copy_previous_rir(imp, row_idx_2, nsamples);
                   }
               }
           }
           else {
               copy_previous_rir(imp, row_idx_1, nsamples);
           }

           // Calculate new output sample
           for (int conv_idx = 0; conv_idx < nsamples; conv_idx++) {
               if (sample_idx-conv_idx >= 0) {
                   int tmp_imp_idx = mod(row_idx_1-conv_idx,nsamples);
                   out[mic_idx + no_mics*sample_idx] += imp[tmp_imp_idx + nsamples*conv_idx] * in[sample_idx - conv_idx];
               }
           }
       }
   }

   delete[] angle;
   delete[] angles;
   delete[] hanning_window;
   delete[] LPI;
   delete[] r;
   delete[] s;
   delete[] L;
   delete[] n;
   delete[] imp;
}


// Kernel funciton
void kernel(const long signal_length, const double *in, const double fs, double *out) {
   if(threadIdx.x >= ROOM_WIDTH || threadIdx.y >= ROOM_DEPTH)
       return;

   // Compute receivers' and the source's positions
   double rr[NO_MICS][3] = {
           {threadIdx.x + 0.4, threadIdx.y + 0.4, 1.5},{threadIdx.x + 0.4, threadIdx.y + 0.6, 1.5},
           {threadIdx.x + 0.6, threadIdx.y + 0.4, 1.5},{threadIdx.x + 0.6, threadIdx.y + 0.6, 1.5}};
   double ss[3] = {blockIdx.x + 0.5, blockIdx.y + 0.5, 1.5};

   // Compute address for result
   int blockId = blockIdx.y + blockIdx.x * gridDim.y;
   int threadId = blockId * (blockDim.x * blockDim.y)
                  + (threadIdx.x * blockDim.y) + threadIdx.y;
   double *res = out + threadId * signal_length * NO_MICS;
   printf("%d th thread launched. Result offset is %d\n", threadId, threadId * signal_length * NO_MICS);
   mexFunction(signal_length, in, fs, (double *)rr, ss, res);
}

int main() {
   int fs;
   int len;
   double *in;

   // Read audio signal
   FILE *fp = fopen("../matlab/input.dat", "r");
   assert(fp != NULL);
   fscanf(fp, "%d\n", &fs);
   fscanf(fp, "%d\n", &len);
   in = new double[len];
   for(int i = 0; i < len; i++)
       fscanf(fp, "%lf\n", in + i);
   fclose(fp);

   // Malloc space for result
   double *out = NULL;
   const int NUM_OF_SIGNALS = 1;
   checkCudaErrors(cudaMallocManaged ((void **) &out, sizeof(double) * NO_MICS * len * NUM_OF_SIGNALS));
   checkCudaErrors(cudaDeviceSynchronize());

   // Determine blocksPerGrid and threadPerBlock
   dim3 blocksPerGrid(9, 9);
   dim3 threadPerBlock(9, 9);

   kernel<<<1, 1>>>(len, in, fs, out);
   //mexFunction<<<blocksPerGrid，threadPerBlock>>>(len, in, fs, out);

   checkCudaErrors(cudaDeviceSynchronize());

   fp = fopen("output.dat", "w");
   for(int k = 0; k < NUM_OF_SIGNALS; k++){
       for(int i = 0; i < len; i++){
           for(int j = 0; j < NO_MICS; j++){
               fprintf(fp, "%lf ", out[k*len*NO_MICS + i*NO_MICS + j]);
           }
           fprintf(fp, "\n");
       }
   }
   fclose(fp);

   return 0;
}


