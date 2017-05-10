/*-BP_FIRNET3.C --------------------------------------------------------------

     This code is for training a three layer network (all layers with taps). 
  Either stochastic (iterative) or batch (gradient) updates can be specified.
  Because there are tapped delays internal to the network, the learning 
  algorithm was adjusted to take these delays into account.

  Eric Wan, Alex Nelson, Thyagarajan Srinivasan  (9/95)
  ericwan@eeap.ogi.edu


Structure of Matlab call :

[W1L,B1L,W2L,B2L,W3L,B3L,MSE] =
                bp_firnet3(W1R,B1R,W2R,B2R,W3R,B3R,InpVects,TargVects,Params);

INPUTS:
-------

W?R: mxn  (Initial Weight Matrix)   m: nodes in the layer
                                    n: nodes in the previous layer
                          W[i,j] is the weight connecting the ith node in the 
			  current layer to the jth node in the previous layer.
			  The "R" denotes "right side," or pretraining weights.

B?R: mx1  (Initial Bias Matrix)     m: nodes in the layer
                          B[i] is the bias weight for the ith node in the 
			  layer.  The "R" denotes "right side," or pretraining 
                          weights.


InpVects: numIn x K (Input Vectors) K: number of input vectors in sequence
                                      numIn: number of network inputs
                          InpVects[i,j] is the ith input from the jth input 
                          vector. Column vector j contains the values of all 
			  of the input series at time k=j. 
			  Each row of the matrix is comprised of an individual 
			  time series, from k=0 to k=K-1. If there are t taps 
                          in the first hidden layer, then each node in that 
		          layer will take values from t columns at the same
       			  time.

TargVects: numOut x K (Target Vectors) K: number of target-output vectors
                                       numOut: number of network outputs
			  TargVects[i,j] is the ith target (desired) output at 
			  time k=j+1. Hence, the jth column vector contains all
			  the target outputs at time k=j+1. Each row of the 
			  matrix is comprised of an individual time series of 
			  target output values, from k=1 to k=K. The number of 
			  columns equals the number of input vectors, but
			  the 0th column of TargVects reflects data at time 
			  k=1, whereas the 0th column of InpVects reflects data
			  at time k=0. Similarly, the Kth target vector 
			  contains data at k=K, whereas the Kth input vector
			  contains data at k=K-1. Also, notice that the first 
			  T-1 target vectors will not be utilized, where T is 
			  the number of taps.

Params: 1 x 4  (Parameter Vector)
                          Params[0] = number of training epochs, where
			  each epoch consists of a pass through all training
			  vectors.
			  Params[1] = the step size or training rate, mu
			  Params[2] = 0 for iterative updates (stochastic)
			            = 1 for block updating (true gradient)
		          Params[3] = the display period for the MSE, given in
			  number of epochs.

OUTPUTS: 
--------
W?L: mxn (Final Weight Matrix)
                          Same as W?R, but contains the weights after the last
			  training epoch.
B?L: mx1 (Final Bias Matrix)
                          Same as B?R, but contains the biases after the last
			  training epoch.
MSE: numOut x E (Mean Squared Error) E: number of training epochs
                                     numOut: number of network outputs
			  MSE[i,j] is the MSE for the ith output node after the
			  jth training epoch. (An epoch consists of the 
			  presentation of all training vectors once.)

	Author : Thyagarajan Srinivasan,
		 Dept. of Electrical Engineering & Applied Physics,
		 Oregon Graduate Institute.
		 thyagu@eeap.ogi.edu
		 Sep 23 1995
-----------------------------------------------------------------------------*/
				/* INCLUDE FILES  */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mex.h"
     
				/* MACROS */
#define FN1(x) (tanh(x))	/* first hidden layer transfer function */
#define FN2(x) (tanh(x))	/* second hidden layer transfer function */
#define FN3(x) (x)		/* output layer transfer function  */
#define dFN1(x) (1.0 - x*x)	/* derivatives of transfer functions */
#define dFN2(x) (1.0 - x*x)
#define dFN3(x) (1)
				/* The following Macros convert the matrix 
				   indices (i,j) into a form useable by the 
				   MATLAB matrix format. This is required 
				   because the matrices used, e.g. W1 and W2, 
				   are infact pointers to a MATLAB object. */
#define c2MAT(i,j,ROWS) (i + j*ROWS)
#define c2MAT_tap(i,j,taps,ROWS,rowsXcolms) (i+ j*ROWS + taps*rowsXcolms)

int numIn;			/* num of input nodes in network */
int numOut;			/* num of output nodes in network */
int numHid1;			/* num of nodes in 1st intermediate layer */
int numHid2;			/* num of nodes in 2nd intermediate layer */
double mu;			/* the learning rate */
long numEpochs;			/* num of training passes */
int displayPeriod;		/* interval between updates of MSE to screen */
double *W1R =NULL;		/* First layer Weights - pre_training */
double *B1R =NULL;		/* First layer Bias - pre_training */
double *W2R =NULL;		/* Second Layer Wieghts - pre_training */
double *B2R =NULL;		/* Second Layer Bias - Pre_training */
double *W3R =NULL;		/* Third Layer Wieghts - pre_training */
double *B3R =NULL;		/* Third Layer Bias - Pre_training */
double *W1L =NULL;		/* First layer Weights - post_training */
double *B1L =NULL;		/* First layer Bias - post_training */
double *W2L =NULL;		/* Second Layer Weights - post_training */
double *B2L =NULL;		/* Second Layer Bias - Post_training */
double *W3L =NULL;		/* Third Layer Weights - post_training */
double *B3L =NULL;		/* Third Layer Bias - Post_training */
double *W1change;		/* change in hidden unit weights */
double *B1change;		/* change in hidden unit biases */
double *W2change;		/* change in hidden unit weights */
double *B2change;		/* change in hidden unit biases */
double *W3change;		/* change in output unit weights */
double *B3change;		/* change in output unit biases */
double *InpVects =NULL;		/* Training Vectors - Input */
double *TargVects =NULL;	/* Training Vectors - Target */
double *netOut=NULL;		/* network output */
double **hid1Out =NULL;		/* Output at intermediate layer [node][tap]*/
double **hid2Out =NULL;		/* Output at second layer [node][tap] */
double *deltaHid1=NULL;		/* delta at the first layer [node][tap]*/
double **deltaHid2=NULL;	/* delta at the second layer [node][tap] */
double **deltaOut=NULL;		/* Network error [node][tap]*/
int numInpVects;		/* the number of training vectors */
int numTaps1;			/* the num of taps in 1st layer*/
int numTaps2;			/* the num of taps in 2nd layer*/
int numTaps3;			/* the num of taps in 3rd layer */
double *cumSqErr= NULL;		/* cumulative sq. err of all output nodes */
double *MSE;			/* the MSE for each training pass */
int trMode;			/* training mode (0=stochastic, 1=Batch) */
int mask2;			/* used for circular indexing of layer 2 */
int mask3;			/* used for circular indexing of layer 3 */

/*
void InitGlobals(int nlhs,Matrix *plhs[],int nrhs,Matrix *prhs[]);
*/
void train(void);
void fwdPassLayer1(int trg_index);
void fwdPassLayer2(int trg_index);
void fwdPassLayer3(int trg_index);
void backpropLayer1(int index);
void backpropLayer2(int index);
void backpropLayer3(int index);
void weight_update(int index);
void calcWeightUpdate(int index);
void batchWeightUpdate(void);
void calc_MSE(int pass);
void resetDeltas();


/* Gateway Routine ----------------------------------------------------------*/

void mexFunction(
		 int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
/*
  InitGlobals(nlhs,plhs,nrhs,prhs);
*//* InitGlobals -------------------------------------------------------------- 
   First loads the values defining the architecture into the global variables, 
  while making sure these values are self-consistant. Prints error messages
  regarding any inconsistancies. 
   Next, assigns global pointers to the MATLAB objects. 
  Finally, allocates memory for the hidden unit outputs and the network 
  outputs, and assigns global pointers to these new objects. */
/*
void InitGlobals(int nlhs, Matrix *plhs[],
		int nrhs, Matrix *prhs[])
{
*/
  int i,colms,rows;		/* num of colms and rows of various matrices */
  int numWeights;

 /* CHECK FOR ERRORS IN ARGUMENTS */
  if(nrhs!=9)
    mexErrMsgTxt("9 Arguments needed on RHS: \
 W1, B1, W2, B2, W3, B3, InpVects, TargVects, Params.\n");
  if(nlhs!=7)
    mexErrMsgTxt("7 Arguments needed on LHS: \
 W1, B1, W2, B2, W3, B3, MSE.\n");
  
  numIn = mxGetM(prhs[6]);	/* num of input nodes = num rows in InpVects */
  numHid1 = mxGetM(prhs[0]);	/* num of hidden nodes = num rows of W1 */

  colms = mxGetN(prhs[0]);	/* number of colms of W1 */
  if((colms % numIn) != 0)	/* ...should be multiple of numIn */
    mexErrMsgTxt("Number of columns of W1 & rows of InpVects incompatible\n");
  numTaps1 = colms /numIn;	/* numTaps betw Inp & Hid is that multiple */

  colms = mxGetN(prhs[2]);	/* number of colms of W2*/
  if((colms % numHid1) !=0)	/* ...should be multiple of numHid1 */
    mexErrMsgTxt("Matrices W1 and W2 are incompatible!\n");
  numTaps2 = colms / numHid1;	/* num of taps betw Hid & Out is that multpl */

  numHid2 = mxGetM(prhs[2]);	/* num of rows of W2 is size of 2nd hid layer*/
  colms = mxGetN(prhs[4]);	/* num of colms of W3 */
  if((colms % numHid2) !=0)	/* ...should be multiple of numHid2*/
    mexErrMsgTxt("Matrices W2 and W3 are incompatible!\n");
  numTaps3 = colms / numHid2;	/* numTaps3 is the multiplier */

  numOut = mxGetM(prhs[4]);	/* number of outputs = num rows of W3 */

  rows = mxGetM(prhs[1]);	/* number of rows of B1 */
  colms =mxGetN(prhs[1]);	/* number of colms of B1 */
  if(rows!=numHid1)
    mexErrMsgTxt("B1 must have the same number of rows as W1\n");
  if(colms!=1)
    mexErrMsgTxt("B1 must have only one column.\n");

  rows = mxGetM(prhs[3]);	/* number of rows of B2 */
  colms =mxGetN(prhs[3]);	/* number of colms of B2 */
  if(rows!=numHid2)
    mexErrMsgTxt("B2 must have the same number of rows as W2\n");
  if(colms!=1)
    mexErrMsgTxt("B2 must have only one column.\n");

  rows = mxGetM(prhs[5]);	/* number of rows of B3 */
  colms =mxGetN(prhs[5]);	/* number of colms of B3 */
  if(rows!=numOut)
    mexErrMsgTxt("B3 must have the same number of rows as W3\n");
  if(colms!=1)
    mexErrMsgTxt("B3 must have only one column.\n");

  numInpVects = mxGetN(prhs[6]); /* the num of input vectors */
  if(numInpVects <= (numTaps1 + numTaps2 + numTaps3))
    mexErrMsgTxt("Error! The number of input vectors must be greater \
than the total number of taps!\n");
  
  colms = mxGetN(prhs[7]);	/* the num of target vectors */
  if(colms != numInpVects)
    mexErrMsgTxt("Unequal Number of Input and Target Vectors\n");

  rows= mxGetM(prhs[7]);	/* length of the target vectors */
  if( rows!=numOut)
    mexErrMsgTxt("Number of rows in Target Vector Matrix must equal \
the number of output nodes!\n");

 /* LOAD TRAINING PARAMETERS */
  numEpochs = mxGetPr(prhs[8])[0]; /* number of training passes */
  if(numEpochs < 1)
    mexErrMsgTxt("Must have at least one Training Epoch\n"); 

  mu = mxGetPr(prhs[8])[1]; /* Learning rate */
  if(mu <= 0)
    mexErrMsgTxt("Learning rate must be positive\n");

  trMode = mxGetPr(prhs[8])[2]; /* unused */
  if((trMode!=0)&&(trMode!=1))
     mexErrMsgTxt("Training Mode must be 0(stochastic) or 1(batch)");

  displayPeriod = mxGetPr(prhs[8])[3]; /*frequency of MSE screen update*/

 /* FORM CIRCULAR INDICES */
  if((numTaps2+numTaps3) > 2)
  {
    mask2= 1 << ((int) (log((double) (numTaps2+numTaps3-2)) / 0.69314718) + 1);
    mask2-=1;
  }
  else
  mask2 = 0;

  if(numTaps3 > 1)
  {
    mask3= 1 << ((int) (log((double) (numTaps3 - 1)) / 0.69314718) + 1);
    mask3-=1;
  }
  else
  mask3 = 0;

 /* ALLOCATE MEMORY AND ASSIGN POINTERS */
  allocReturnMatrices(plhs);	   /* Create Space for the return Argvuments */
  assignMatrixPointers(plhs,prhs); /* Assign pointers to internal variables */
  allocInternalMemory();	   /* Create Space for internal variables   */

 /* COPY INITIAL WEIGHTS INTO TRAINING WEIGHT MATRICES */
 numWeights = numOut*numHid2* numTaps3;
  for(i=0;i<numWeights;i++)
    W3L[i]=W3R[i];
  for(i=0;i<numOut;i++)
    B3L[i]=B3R[i];

  numWeights = numHid2*numHid1* numTaps2;
  for(i=0;i<numWeights;i++)
    W2L[i]=W2R[i];
  for(i=0;i<numHid2;i++)
    B2L[i]=B2R[i];

  numWeights = numHid1 * numIn * numTaps1 ;
  for(i=0;i<numWeights;i++)
    W1L[i]= W1R[i];
  for(i=0;i<numHid1;i++)
    B1L[i]= B1R[i];

  if(displayPeriod > 0)
  {  printf("Inputs: %d\t1st Hid. Layer: %d\t2nd Hid. Layer: %d\tOutputs: %d\n",
	 numIn,numHid1,numHid2,numOut);
    printf("1st Layer Taps: %d\t2nd Layer Taps: %d\t3rd Layer Taps: %d\n",
	 numTaps1,numTaps2,numTaps3);
    printf("Input Vectors: %d \tEpochs: %ld\n",numInpVects, numEpochs);
   }
/*
}
*/

 
    train();
}

/* Train ---------------------------------------------------------------------
   Check the training mode to determine whether to do stochastic or batch 
 training. Runs through specified number of epochs, where each epoch consists
 of the presentation of all training vectors in InpVects. Each input vector
 is passed forward through the network, and its consequent error signals are
 backpropagated to produce delta values. In stochastic mode, these deltas are
 immediately used to change the weights. In batch mode, the weight changes are
 calculated and accumulated for use at the end of each epoch. In either 
 training mode, the MSE of each output unit is calculated at the end of the 
 epoch. */
void train(void)
{ int epoch,i,j,tmp;
  int hlf_vcts;			/* half the num of training vectors */

  if(trMode == 0)		/* IF STOCHASTIC Mode */
  {  for(epoch=0;epoch<numEpochs;epoch++)
     {
       for(i=numTaps1-1;i<(numTaps1+numTaps2-2);i++)
	 fwdPassLayer1(i);	/* prime second layer of taps */
       for(;i< (numTaps1+numTaps2+numTaps3-3);i++)
	 {
	   fwdPassLayer1(i);	/* prime third layer of taps */
	   fwdPassLayer2(i);
	 }
       for(;i<(numTaps1+numTaps2+2*numTaps3-4);i++)
       {
	 fwdPassLayer1(i);	/* prime the output-node deltas */
	 fwdPassLayer2(i);
	 fwdPassLayer3(i);
	 backpropLayer3(i);
	 weight_update(i);	/* adjust 3rd layer weights */
       }
       for(;i< (numTaps1+2*numTaps2+2*numTaps3-5);i++)
       {
	 fwdPassLayer1(i);	/* prime the 2nd hidden layer deltas */
	 fwdPassLayer2(i);
	 fwdPassLayer3(i);
	 backpropLayer3(i);
	 backpropLayer2(i);
	 weight_update(i);	/* adjust the 3rd & 2nd layer weights */
       }
       for(;i<numInpVects;i++)	/* train over the remaining data */
       { 
	 fwdPassLayer1(i);
	 fwdPassLayer2(i);
	 fwdPassLayer3(i);
	 backpropLayer3(i);
	 backpropLayer2(i);
	 backpropLayer1(i);
	 weight_update(i);
       }
       for(;i<numInpVects+numTaps3+numTaps2-2;i++)
       {
	 tmp = i& mask3;
	 for(j=0;j<numOut;j++)	/* flush the output deltas */
	   deltaOut[j][tmp] =0.0;
	 backpropLayer2(i);
	 backpropLayer1(i);
	 weight_update(i);
       }
       resetDeltas();		/* clear all deltas to zero */
       calc_MSE(epoch);
     }
   }
  if(trMode == 1)		/* IF BATCH Mode */
  {  for(epoch=0;epoch<numEpochs;epoch++)
     {
       for(i=numTaps1-1;i<(numTaps1+numTaps2-2);i++)
	 fwdPassLayer1(i);	/* prime second layer of taps */
       for(;i< (numTaps1+numTaps2+numTaps3-3);i++)
	 {
	   fwdPassLayer1(i);	/* prime third layer of taps */
	   fwdPassLayer2(i);
	 }
       for(;i<(numTaps1+numTaps2+2*numTaps3-4);i++)
       {
	 fwdPassLayer1(i);	/* prime the output-node deltas */
	 fwdPassLayer2(i);
	 fwdPassLayer3(i);
	 backpropLayer3(i);
	 calcWeightUpdate(i);	/* adjust 3rd layer weights */
       }
       for(;i< (numTaps1+2*numTaps2+2*numTaps3-5);i++)
       {
	 fwdPassLayer1(i);	/* prime the 2nd hidden layer deltas */
	 fwdPassLayer2(i);
	 fwdPassLayer3(i);
	 backpropLayer3(i);
	 backpropLayer2(i);
	 calcWeightUpdate(i);	/* adjust the 3rd & 2nd layer weights */
       }
       for(;i<numInpVects;i++)	/* train over the remaining data */
       { 
	 fwdPassLayer1(i);
	 fwdPassLayer2(i);
	 fwdPassLayer3(i);
	 backpropLayer3(i);
	 backpropLayer2(i);
	 backpropLayer1(i);
	 calcWeightUpdate(i);
       }
       for(;i<numInpVects+numTaps3+numTaps2-2;i++)
       {
	 tmp = i& mask3;
	 for(j=0;j<numOut;j++)	/* flush the output deltas */
	   deltaOut[j][tmp] =0.0;
	 backpropLayer2(i);
	 backpropLayer1(i);
	 calcWeightUpdate(i);
       }
       batchWeightUpdate();
       resetDeltas();		/* clear all deltas to zero */
       calc_MSE(epoch);		/* compute MSE for each output node */
     }
  }
  if(((numEpochs % displayPeriod) != 0.0) && (displayPeriod > 0))
  {  printf("Epoch: %ld",numEpochs);
     for(i=0;i<numOut;i++)
       printf(" %lf",MSE[c2MAT(i,(numEpochs-1),numOut)]);
     printf("\n");
  }
}

/* Foward Pass Layer 1 ----------------------------------------------------
  Takes the (numTaps1) most recent input vectors and propagates the signal 
 forward through the first set of taps and weights for one time step to 
 produce an output vector for the first hidden layer. */
void fwdPassLayer1(int trg_index)
{
int j,k,t,index,tmp,tmp1;
double fin_out;
int numHid1XnumIn = numHid1*numIn;

  tmp1 = trg_index & mask2;
  for(k=0;k<numHid1;k++)	/* for each node in the layer */
  {
    fin_out=B1L[k];		/* the bias */
    for(t=0;t<numTaps1;t++)
    {
      tmp = trg_index-t;
      for(j=0;j<numIn;j++)	/* weighted sum of inputs */
        fin_out+= InpVects[c2MAT(j,tmp,numIn)] *
	          W1L[c2MAT_tap(k,j,t,numHid1,numHid1XnumIn)];
    }
    hid1Out[k][tmp1]=FN1(fin_out); /*sigmoid of weighted sum */
  }
}

/* Foward Pass Layer 2 ----------------------------------------------------
  Takes the (numTaps1) most recent input vectors and propagates the signal 
 forward through the second  set of taps and weights for one time step to 
 produce an output vector for the second hidden layer. */
void fwdPassLayer2(int trg_index)
{
int j,k,t,tmp,tmp1;
double fin_out;
int numHid2XnumHid1 = numHid2*numHid1;

  tmp1 = trg_index & mask3;

  for(k=0;k<numHid2;k++)
  {
    fin_out=B2L[k];
    for(t=0;t<numTaps2;t++)
    {
      tmp = (trg_index-t) & mask2;
      for(j=0;j<numHid1;j++)
        fin_out += hid1Out[j][tmp] * 
	           W2L[c2MAT_tap(k,j,t,numHid2,numHid2XnumHid1)];
    }
    hid2Out[k][tmp1] = FN2(fin_out);
  }
}

/* Foward Pass Layer 3 ----------------------------------------------------
  Takes the (numTaps1) most recent input vectors and propagates the signal 
 forward through the third set of taps and weights for one time step to 
 produce an output vector for the third hidden layer. */
void fwdPassLayer3(int trg_index)
{
int j,k,t,tmp;
double fin_out;
int numOutXnumHid2 = numOut*numHid2;

  for(k=0;k<numOut;k++)
  {
    fin_out=B3L[k];
    for(t=0;t<numTaps3;t++)
    {
      tmp = (trg_index-t) & mask3;
      for(j=0;j<numHid2;j++)
        fin_out+=hid2Out[j][tmp] * W3L[c2MAT_tap(k,j,t,numOut,numOutXnumHid2)];
    }
    netOut[k] = FN3(fin_out);
  }
}

/* Backpropagate Layer 3 ----------------------------------------------------
   Computes the errors in the output layer (and adds them to the cumulative
  squared errors), then copies these errors to form the delta values
  in the output nodes */
void backpropLayer3(int index)
{
int i,j,tmp;
double err;
  
  tmp = index & mask3;
  for(i=0;i<numOut;i++)
  {
    err= TargVects[c2MAT(i,index,numOut)] - netOut[i];
    cumSqErr[i] += err * err;
    deltaOut[i][tmp]=err;
  }
}

/* Backpropagate Layer 2 ----------------------------------------------------
   Takes the errors in the output layer, and uses them to compute the delta
 values in the second layer of hidden nodes */
void backpropLayer2(int index)
{
int i,j,t,tmp,tmp1;
double delta;
int numOutXnumHid2 = numOut*numHid2;

  tmp1 = (index - numTaps3 + 1 ) & mask3;
  for(i=0;i<numHid2;i++)
  {
    delta=0.0;
    for(t=0;t<numTaps3;t++)
    {
      tmp = (index - t) & mask3;
      for(j=0;j<numOut;j++)
        delta += deltaOut[j][tmp] *
	         W3L[c2MAT_tap(j,i,(numTaps3-1-t),numOut,numOutXnumHid2)];
    }
    deltaHid2[i][index & mask2]= delta * dFN2(hid2Out[i][tmp1]);
  }
}

/* Backpropagate Layer 1 ----------------------------------------------------
   Takes the errors in the second hidden layer, and uses them to compute the 
 delta values in the first layer of hidden nodes */
void backpropLayer1(int index)
{
int i,j,t,tmp,tmp2;
double delta;
int numHid2XnumHid1 = numHid2*numHid1;

  tmp2 = (index -numTaps2-numTaps3+2 ) & mask2;
  for(i=0;i<numHid1;i++)
  {
    delta=0.0;
    for(t=0;t<numTaps2;t++)
    {
      tmp = (index -t) & mask2;
      for(j=0;j<numHid2;j++)
        delta += deltaHid2[j][tmp] *
	         W2L[c2MAT_tap(j,i,(numTaps2-1-t),numHid2,numHid2XnumHid1)];
    }
    deltaHid1[i]= delta * dFN1(hid1Out[i][tmp2]);
  }
}

/* Weight Update ------------------------------------------------------------
  For Stochastic (Iterative) backprop training. Changes all network weights
 based on the delta values computed during backpropagation. */
void weight_update(int index)
{
  int i,j,t;
  int tmp,tmp1;
  double mu_del;
  int index1, index2;
  int numOutXnumHid2 = numOut*numHid2;
  int numHid2XnumHid1 = numHid2*numHid1;
  int numHid1XnumIn = numHid1*numIn;

  for(i=0;i<numOut;i++)		/* FOR OUTPUT Nodes */
  {
    mu_del = mu * deltaOut[i][index & mask3];
    for(t=0;t<numTaps3;t++)
    {
      tmp = (index -t) & mask3;
      for(j=0;j<numHid2;j++)
        W3L[c2MAT_tap(i,j,t,numOut,numOutXnumHid2)] += mu_del * 
	                                               hid2Out[j][tmp];
    }
    B3L[i] += mu_del;
  }

  tmp1 = index - numTaps3+1;
  for(i=0;i<numHid2;i++)	/* FOR Second HIDDEN Layer Nodes */
  {
    mu_del = mu * deltaHid2[i][index & mask2];
    for(t=0;t<numTaps2;t++)
    {
      tmp = (tmp1 -t) & mask2;
      for(j=0;j<numHid1;j++)
        W2L[c2MAT_tap(i,j,t,numHid2,numHid2XnumHid1)] += mu_del * 
	                                                 hid1Out[j][tmp];
    }
    B2L[i] += mu_del;
  }

  tmp1 = index -numTaps2-numTaps3+2;
  for(i=0;i<numHid1;i++)	/* FOR First HIDDEN Layer Nodes */
  {
    mu_del = mu * deltaHid1[i];
    for(t=0;t<numTaps1;t++)
    {
      tmp =tmp1 -t;
      for(j=0;j<numIn;j++)
        W1L[c2MAT_tap(i,j,t,numHid1,numHid1XnumIn)] += mu_del * 
                                                  InpVects[c2MAT(j,tmp,numIn)];
    }
    B1L[i] += mu_del;
  }
}
 
/* Calc Weight Update ---------------------------------------------------------
  Used in batch mode only. Accumulates the weight changes during the training
 epoch in order to approximate the direction of gradient decent. The weight
 changes are not used until the end of the training epoch.*/
void calcWeightUpdate(int index)
{
  int i,j,t;
  int tmp,tmp1;
  int index1, index2;
  int numOutXnumHid2 = numOut*numHid2;
  int numHid2XnumHid1 = numHid2*numHid1;
  int numHid1XnumIn = numHid1*numIn;

  for(i=0; i<numOut; i++)
  { for(t=0; t<numTaps3; t++)
    {
      tmp = (index - t) & mask3;
      for(j=0;j<numHid2;j++)
        W3change[c2MAT_tap(i,j,t,numOut,numOutXnumHid2)] += hid2Out[j][tmp] * 
	                                            deltaOut[i][index & mask3];
    }
    B3change[i] += deltaOut[i][index & mask3];
  }

  tmp1 = index - numTaps3 + 1;
  for(i=0; i<numHid2; i++)
  { for(t=0; t<numTaps2; t++)
    {
      tmp = (tmp1 - t) & mask2;
      for(j=0; j<numHid1; j++)
        W2change[c2MAT_tap(i,j,t,numHid2,numHid2XnumHid1)] += hid1Out[j][tmp] *
	                                         deltaHid2[i][index & mask2];
    }
    B2change[i] += deltaHid2[i][index & mask2];
  }

  tmp1 = index - numTaps2 - numTaps3 + 2;
  for(i=0; i<numHid1; i++)
  { for(t=0; t<numTaps1; t++)
    {
      tmp = tmp1 - t;
      for(j=0; j<numIn; j++)
        W1change[c2MAT_tap(i,j,t,numHid1,numHid1XnumIn)] += deltaHid1[i] *
                                                  InpVects[c2MAT(j,tmp,numIn)];
    }
    B1change[i] += deltaHid1[i];
  }
}

/* Batch Weight Update --------------------------------------------------------
  Effects the weight changes calculated by calcWeightUpdate. Called at the end
 of the training epoch when using batch mode.*/
void batchWeightUpdate()
{
  int i,j,t;
  int tmp,tmp1;
  int mindex;
  int numOutXnumHid2 = numOut*numHid2;
  int numHid2XnumHid1 = numHid2*numHid1;
  int numHid1XnumIn = numHid1*numIn;

  for(i=0;i<numOut;i++)		/* OUTPUT WEIGHTS */
  { for(t=0;t<numTaps3;t++)
    { for(j=0;j<numHid2;j++)
      { mindex = c2MAT_tap(i,j,t,numOut,numOutXnumHid2);
	W3L[mindex] += W3change[mindex] * mu;
	W3change[mindex] = 0.0;
      }
    }
    B3L[i] += B3change[i] * mu;
    B3change[i] = 0.0;
  }

  for(i=0;i<numHid2;i++)	/* 2ND HIDDEN WEIGHTS */
  { for(t=0;t<numTaps2;t++)
    { for(j=0;j<numHid1;j++)
      { 
	mindex = c2MAT_tap(i,j,t,numHid2,numHid2XnumHid1);
	W2L[mindex] += W2change[mindex] * mu;
	W2change[mindex] = 0.0;
      }
    }
    B2L[i] += B2change[i] * mu;
    B2change[i] = 0.0;
  }

  for(i=0;i<numHid1;i++)	/* 1ST HIDDEN WEIGHTS */
  { for(t=0;t<numTaps1;t++)
    { for(j=0;j<numIn;j++)
      {
	mindex = c2MAT_tap(i,j,t,numHid1,numHid1XnumIn);
	W1L[mindex] += W1change[mindex] * mu;
	W1change[mindex] = 0.0;
      }
    }
    B1L[i] += B1change[i] * mu;
    B1change[i] = 0.0;
  }
}

/* Calc MSE -----------------------------------------------------------------
  Calculates the approximate Mean Squared Error of each network output over
 the current epoch by dividing the cumulative squared error for that output 
 (accumulated over the epoch) by the number of input patterns in the epoch.
   The MSE is stored in an array, and also displayed to the standard output
 if the epoch number is an integer multiple of the display period. */
void calc_MSE(int epoch)
{
int i,j;

  for(i=0;i<numOut;i++)
  {
    MSE[c2MAT(i,epoch,numOut)]= cumSqErr[i] / 
                (double)(numInpVects - numTaps1 - numTaps2 - numTaps3 + 3);
    cumSqErr[i]=0.0;
  }
  if (((epoch+1) % displayPeriod)==0) /* IF it's time to display MSE */
  {
    printf("Epoch: %ld",epoch + 1);
    for(i=0;i<numOut;i++)
      printf(" %lf",MSE[c2MAT(i,epoch,numOut)]);
    printf("\n");
  }
}

/* Reset Deltas -------------------------------------------------------------
  Clears all delta values in the output layer (for all nodes and all taps) */
void resetDeltas()
{
int i,t;

  for(i=0;i<numOut;i++)
  {
    for(t=numTaps2+numTaps3;t<mask3+1;t++)
      deltaOut[i][t]= 0.0;
  }

  for(i=0;i<numHid2;i++)
  {
    for(t=numTaps2;t<=mask2;t++)
      deltaHid2[i][t]=0.0;
  }
}



/* Allocate Return Matrices -------------------------------------------------
  Creates space for the return matrices (MATLAB matrices).*/
allocReturnMatrices(mxArray *plhs[])
{
  plhs[0] = mxCreateDoubleMatrix(numHid1,(numTaps1*numIn), mxREAL); /* creating W1L */
  plhs[1] = mxCreateDoubleMatrix(numHid1,1, mxREAL); /* creating B1L */
  plhs[2] = mxCreateDoubleMatrix(numHid2,(numTaps2 *numHid1), mxREAL); /* creating W2L */
  plhs[3] = mxCreateDoubleMatrix(numHid2,1, mxREAL); /* creating B2L */
  plhs[4] = mxCreateDoubleMatrix(numOut,(numTaps3 *numHid2), mxREAL); /* creating W3L */
  plhs[5] = mxCreateDoubleMatrix(numOut,1, mxREAL); /* creating B3L */
  plhs[6] = mxCreateDoubleMatrix(numOut,numEpochs, mxREAL); /* creating MSE */
}

/* Assign Matrix Pointer -----------------------------------------------------
  Assigns C Pointers to the Matlab matrix arguments.*/
assignMatrixPointers(mxArray *plhs[], mxArray *prhs[])
{
  W1R = mxGetPr(prhs[0]);
  B1R = mxGetPr(prhs[1]);
  W2R = mxGetPr(prhs[2]);
  B2R = mxGetPr(prhs[3]);
  W3R = mxGetPr(prhs[4]);
  B3R = mxGetPr(prhs[5]);
  InpVects = mxGetPr(prhs[6]);
  TargVects = mxGetPr(prhs[7]);
  W1L = mxGetPr(plhs[0]);
  B1L = mxGetPr(plhs[1]);
  W2L = mxGetPr(plhs[2]);
  B2L = mxGetPr(plhs[3]);
  W3L = mxGetPr(plhs[4]);
  B3L = mxGetPr(plhs[5]);
  MSE = mxGetPr(plhs[6]);
}

/* Allocate Internal Memory ---------------------------------------------------
  Creates space for data internal to the program, such as hidden node 
 output values, weight changes, and MSEs. */
allocInternalMemory(void)
{ int i;
 
  hid1Out=(double **) mxCalloc(numHid1 , sizeof(double *));
  if(hid1Out == NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  for(i=0;i<numHid1;i++)
  {
    hid1Out[i]= (double *) mxCalloc ((mask2 +1) , sizeof(double));
    if(hid1Out[i]==NULL)
      mexErrMsgTxt("Unable to Allocate memory!\n");
  }

  hid2Out=(double **) mxCalloc(numHid2 , sizeof(double *));
  if(hid2Out == NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  for(i=0;i<numHid2;i++)
  {
    hid2Out[i]= (double *) mxCalloc ((mask3+1) , sizeof(double));
    if(hid2Out[i]==NULL)
      mexErrMsgTxt("Unable to Allocate memory!\n");
  }

  netOut=(double *) mxCalloc(numOut , sizeof(double));
  if(netOut ==NULL)
    mexErrMsgTxt("Unable to Allocate Memory !\n");

  deltaHid1 = (double *) mxCalloc (numHid1 , sizeof(double));
  /* delta in the first layer */
  if(deltaHid1==NULL)
    mexErrMsgTxt("Unable to Allocate Memory !\n");

  deltaHid2 = (double **) mxCalloc (numHid2 , sizeof(double *));
  if(deltaHid2==NULL)
    mexErrMsgTxt("Unable to Allocate Memory !\n");

  for(i=0;i<numHid2;i++)
  {
    deltaHid2[i]= (double *) mxCalloc ((mask2+1), sizeof(double));
    if(deltaHid2[i] == NULL)
      mexErrMsgTxt("Unable to Allocate Memory !\n");
  }

  deltaOut = (double **) mxCalloc (numOut , sizeof(double *));
  if(deltaOut==NULL)
    mexErrMsgTxt("Unable to Allocate Memory !\n");

  for(i=0;i<numOut;i++)
  {
    deltaOut[i]= (double *) mxCalloc ((mask3+1), sizeof(double));
    if(deltaOut[i] == NULL)
      mexErrMsgTxt("Unable to Allocate Memory !\n");
  }

  W1change = (double *) mxCalloc(numIn*numHid1*numTaps1, sizeof(double));
  if(W1change == NULL)
  mexErrMsgTxt("Unable to Allocate Memory!\n");

  B1change = (double *) mxCalloc(numHid1, sizeof(double));
  if(B1change == NULL)
  mexErrMsgTxt("Unable to Allocate Memory!\n");

  W2change = (double *) mxCalloc(numHid1*numHid2*numTaps2, sizeof(double));
  if(W2change == NULL)
  mexErrMsgTxt("Unable to Allocate Memory!\n");

  B2change = (double *) mxCalloc(numHid2, sizeof(double));
  if(B2change == NULL)
  mexErrMsgTxt("Unable to Allocate Memory!\n");

  W3change = (double *) mxCalloc(numHid2*numOut*numTaps3, sizeof(double));
  if(W2change == NULL)
  mexErrMsgTxt("Unable to Allocate Memory!\n");

  B3change = (double *) mxCalloc(numOut, sizeof(double));
  if(B2change == NULL)
  mexErrMsgTxt("Unable to Allocate Memory!\n");

  cumSqErr=(double *) mxCalloc (numOut , sizeof(double));
  if(cumSqErr ==NULL)
    mexErrMsgTxt("Unable to Allocate Memory !\n");
}
