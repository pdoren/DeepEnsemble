/*----------------------------------------------------------------------------
 This code is for training a two layer neural network, wherein the nodes in the
first (hidden) layer can have multiple taps. That is, the past values of each 
input are used to compute the outputs of the hidden layer nodes. Because the 
output nodes do not have taps, the standard backpropogation algorithm is 
employed. Training can be done in either stochastic mode or batch mode.

Eric Wan, Alex Nelson, Thyagarajan Srinivasan  (9/95)
ericwan@eeap.ogi.edu


Structure of Matlab call :

   [W1L,B1L,W2L,B2L,MSE] = bp_train(W1R,B1R,W2R,B2R,InpVects,TargVects,Params);

INPUTS:
-------

W?R: mxn  (Initial Weight Matrix)   m: nodes in the layer
                                    n: nodes in the previous layer
                          W[i,j] is the weight connecting the ith node in the 
			  current layer to the jth node in the previous layer.

B?R: mx1  (Initial Bias Matrix)     m: nodes in the layer
                          B[i,1] is the bias weight for the ith node in the 
			  layer.

InpVects: numIn x K (Input Vectors) K: number of input vectors in sequence
                                      numIn: number of network inputs
                          InpVects[i,j] is the ith input from the jth input 
			  vector. Column vector j contains the values of all 
			  of the input series at time k=j. 
			  Each row of the matrix is comprised of an individual 
			  time series, from k=0 to k=K-1. If there are t taps 
			  in the hidden layer, then each hidden node will take 
			  values from t columns 'simultaneously'.

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
		 Sep 13 1995
-----------------------------------------------------------------------------*/
				/* INCLUDE FILES  */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mex.h"

				/* MACROS */
#define FN1(x) (tanh(x))	/* hidden layer transfer function */
#define FN2(x) (x)		/* output layer transfer function */
#define dFN1(x) (1.0 -(x*x))	/* derivative of FN1 wrt x */
#define dFN2(x) (1)		/* derivative of FN2 wrt x */
				/* The following Macros convert the matrix 
				   indices (i,j) into a form useable by the 
				   MATLAB matrix format. This is required 
				   because the matrices used, e.g. W1 and W2, 
				   are infact pointers to a MATLAB object. */
#define c2MAT(i,j,ROWS) (i + j*ROWS)
#define c2MAT_tap(i,j,taps,ROWS,rowsXcolms) (i+ j*ROWS + taps*rowsXcolms)

			     /* GLOBAL VARIABLES */
int numIn;			/* # of input nodes in network */
int numOut;			/* # of output nodes in network */
int numHid;			/* # nodes in inetermediate layer */
double mu;			/* the learning rate */
long numEpochs;			/* # of training epochs */
int displayPeriod;		/* time between display of MSE */
double *W1R =NULL;		/* First layer Weights - pre_training */
double *B1R =NULL;		/* First layer Bias - pre_training */
double *W2R =NULL;		/* Second Layer Wieghts - pre_training */
double *B2R =NULL;		/* Second Layer Bias - Pre_training */
double *W1L =NULL;		/* First layer Weights - post_training */
double *B1L =NULL;		/* First layer Bias - post_training */
double *W2L =NULL;		/* Second Layer Wieghts - post_training */
double *B2L =NULL;		/* Second Layer Bias - Post_training */
double *W1change;		/* change in hidden unit weights */
double *B1change;		/* change in hidden unit biases */
double *W2change;		/* change in output unit weights */
double *B2change;		/* change in output unit biases */
double *InpVects =NULL;		/* Training Vectors - Input */
double *TargVects =NULL;	/* Training Vectors - Target */
double *netOut=NULL;		/* pointer to network outputs */
double *hidOut =NULL;		/* Output at intermediate layer */
double *deltaHid=NULL;		/* delta at the first layer */
double *deltaOut=NULL;		/* Network output error */
int numInpVects;		/* the number of training vectors */
int numTaps;			/* the # of taps */
int *randi = NULL;		/* table of randomized vector indices */
double *cumSqErr = NULL;	/* cumulative sq. error (vector) over trg set*/
double *MSE;			/* MSE vectors for each training epoch */
int trMode;			/* training mode (0= stochastic); (1: Batch) */

			     /* FUNCTION PROTOTYPES */
/*
void InitGlobals(int nlhs,mxArray *plhs[],int nrhs,mxArray *prhs[]);
*/
void train(void);
void shuffle(int vectors);
void fwd_pass(int index);
void weight_update(int index);
void calcWeightUpdate(int index);
void batchWeightUpdate(int numVects);
void back_prop(int index);
void calc_MSE(int pass);

/*-------------------------------------------------------------------------*/
/* Gateway Routine */

void mexFunction(
		int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[])
{
/*
   InitGlobals(nlhs,plhs,nrhs,prhs);
*/
    /* InitGlobals -------------------------------------------------------------- 
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

 /* Error Checking */

  if(nrhs!=7)
    mexErrMsgTxt("7 Args needed on RHS: W1,B1,W2,B2,InpVects,TargVects,Params");
  if(nlhs!=5)
    mexErrMsgTxt("5 Arguments needed on LHS: W1, B1, W2, B2, MSE ");
  
  numIn = mxGetM(prhs[4]);	/* num of input nodes = num rows in InpVects */
  numHid = mxGetM(prhs[0]);	/* number of hidden nodes = num rows of W1 */
  colms = mxGetN(prhs[0]);	/* number of colms of W1 */

  if((colms % numIn) != 0)	/* num of colms of W1 shld be mult of numIn */
    mexErrMsgTxt("Invalid W1!\n");
  numTaps = colms /numIn;	/* ... and the num of Taps is that multiple */

  colms = mxGetN(prhs[2]);	/* number of colms of W2... */
  if(colms != numHid)		/* ...should equal the num of Hidden Nodes */
    mexErrMsgTxt("Matrices W1 and W2 are incompatible!\n");
  
  numOut = mxGetM(prhs[2]);	/* the number of outputs = num rows of W2 */
  
  rows = mxGetM(prhs[1]);       /* the number of rows of B1 */
  colms =mxGetN(prhs[1]);	/* the number of colms of B1 */
  if(rows!=numHid)
    mexErrMsgTxt("B1 must have the same number of rows as W1\n");
  if(colms!=1)
    mexErrMsgTxt("B1 must have only one column.\n");
  
  rows = mxGetM(prhs[3]);	/* the number of rows of B2 */
  colms =mxGetN(prhs[3]);	/* the number of colms of B2 */
  if(rows!=numOut)
    mexErrMsgTxt("B2 must have the same number of rows as W2\n");
  if(colms!=1)
    mexErrMsgTxt("B2 must have only one column.\n");
  
  numInpVects=mxGetN(prhs[4]);	/* the number of input vectors */
  if(numInpVects<=numTaps)
  {  mexErrMsgTxt("Error! The number of input vectors must be greater \
than the number of taps!\n");
  }
  rows = mxGetM(prhs[4]);	/* the length of the input vectors */
  if(rows != numIn)
    mexErrMsgTxt("Input vector length must equal number of input nodes.\n");

  colms = mxGetN(prhs[5]);	/* the num of target vectors */
  if(colms != numInpVects)	/* ...should equal the num of input vects */
    mexErrMsgTxt("Must be the same number of Input and Target Vectors\n");
  
  rows= mxGetM(prhs[5]);	/* the num of rows of TargVects... */
  if( rows!=numOut)		/* ...should equal the num of outputs */
  { mexErrMsgTxt("Number of rows in Target Vector Matrix must equal \
the number of output nodes!\n");
  }

 /* Load Training Parameters  */
  numEpochs = mxGetPr(prhs[6])[0]; /* Number of Training Epochs */
  if(numEpochs < 1)
    mexErrMsgTxt("Must have at least one Training Epoch\n"); 

  mu = mxGetPr(prhs[6])[1];	   /* Learning rate */
  if(mu < 0)
     mexErrMsgTxt("Learning rate must be positive\n");

  trMode = mxGetPr(prhs[6])[2];	   /* 0=stochastic , 1=batch training */
  if((trMode!=0)&&(trMode!=1))
     mexErrMsgTxt("Training Mode must be 0(stochastic) or 1(batch)");

  displayPeriod = mxGetPr(prhs[6])[3]; /* num of epochs between MSE displays */

 /* Allocate Memory and Assign Pointers */
  allocReturnMatrices(plhs);	   /* Create Space for the return Arguments */
  assignMatrixPointers(plhs,prhs); /* Assign pointers to internal variables */
  allocInternalMemory();	   /* Create Space for internal variables   */

  for(i=0;i<numInpVects;i++)       /* initialize the table of random indices */
    randi[i]=i;

 /* Copy initial weights into the training weight matrices */
  numWeights = numOut*numHid;
  for(i=0;i<numWeights;i++)
    W2L[i]=W2R[i];
  for(i=0;i<numOut;i++)
    B2L[i]=B2R[i];

  numWeights = numHid * numIn * numTaps ;
  for(i=0;i<numWeights;i++)
    W1L[i]= W1R[i];
  for(i=0;i<numHid;i++)
    B1L[i]= B1R[i];

  if(displayPeriod > 0)
  { printf("\nInputs: %d\nHidden Nodes: %d\nOutput Nodes: %d\n",
	 numIn,numHid,numOut);
    printf("Training Vectors: %d\nTaps: %d Epochs: %ld\n",
	 numInpVects,numTaps,numEpochs);
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
 epoch. In stochastic mode, the order of presentation of training vectors is
 randomized between epochs. */
void train(void)
{ int epoch,i;
  int hlf_vcts;			/* half the # of training vectors */

  if(trMode == 0)		/* IF STOCHASTIC Mode */
  {   hlf_vcts=numInpVects /2;	/* (for shuffling call) */
      for(epoch=0;epoch<numEpochs;epoch++) /* FOR all epochs */
      {
	for(i=numTaps-1;i<numInpVects;i++) /* over all training vectors */
        {
	  fwd_pass(randi[i]);	/* do forward pass through net */
	  back_prop(randi[i]);	/* compute deltas */
	  weight_update(randi[i]);	/* change weights */
	}
	calc_MSE(epoch);	/* compute MSE for each output */
	shuffle(hlf_vcts);	/* shuffle the training vectors */
      }
  }
  else if(trMode == 1)		/* ELSE IF BATCH Mode */
  {  for(epoch=0;epoch<numEpochs;epoch++) /* FOR all epochs */
     {
       for(i=numTaps-1;i<numInpVects;i++) /* over all training vectors */
       {
	 fwd_pass(i);	/* do forward pass through net */
	 back_prop(i);	/* compute deltas */
	 calcWeightUpdate(i); /* calculate incremental weight change */
       }
       batchWeightUpdate(numInpVects);	/* change weights */
       calc_MSE(epoch);	/* compute MSE for each output */
     }
  }
  if(((numEpochs % displayPeriod) != 0.0) && (displayPeriod > 0))
  { printf("Epoch: %ld",numEpochs); /* Print out final MSE values */
    for(i=0;i<numOut;i++)
      printf(" %lf",MSE[c2MAT(i,(numEpochs-1),numOut)]);
    printf("\n");
  }
}

/* Forward Pass ---------------------------------------------------------------
  Takes the most recent numTaps input vectors and propagates the signal forward 
 through the network to produce an output vector. */
void fwd_pass(int index)
{
 int j,k,t;
 int tmp;
 double fin_out;
 int numHidXnumIn = numHid*numIn;

 /* output of first layer */
  for(k=0;k<numHid;k++)		/* FOR each hidden node */
  {
    fin_out=B1L[k];		/*  the bias */
    for(t=0;t<numTaps;t++)	/*  FOR all taps on all inputs */
    {
      tmp = index -t;
      for(j=0;j<numIn;j++)	/*  take weighted sum of inputs */
	fin_out+= InpVects[c2MAT(j,tmp,numIn)] * 
	          W1L[c2MAT_tap(k,j,t,numHid,numHidXnumIn)];
    }
    hidOut[k]=FN1(fin_out);	/* output is sigmoid of weighted sum */
  }
 
 /* output of second layer */
  for(k=0;k<numOut;k++)		/* FOR each output node */
  {
    fin_out=B2L[k];
    for(j=0;j<numHid;j++)	/* compute weighted sum of hid outputs */
      fin_out += hidOut[j] * W2L[c2MAT(k,j,numOut)];
    netOut[k] = FN2(fin_out);	/* output is linear function */
  }
}

/* Backprop -------------------------------------------------------------------
   Computes the errors in the output layer (and adds them to the cumulative
  squared errors), then backpropagates these errors to form the delta values
  in the hidden nodes */
void back_prop(int index)
{
int i,j;
double err,delta;
  
  for(i=0;i<numOut;i++)
  {
    /* error in the output layer */
    err= TargVects[c2MAT(i,index,numOut)] - netOut[i];
    cumSqErr[i]+= err * err;
    deltaOut[i]=err;
  }

  for(i=0;i<numHid;i++)
  {
    delta=0.0;
    /* calculating the back propagated delta */
    for(j=0;j<numOut;j++)
      delta += deltaOut[j] * W2L[c2MAT(j,i,numOut)];
    deltaHid[i]= delta * dFN1(hidOut[i]);
  }
}

/* Weight Update --------------------------------------------------------------
  For Stochastic (Iterative) backprop training. Changes all network weights
 based on the delta values computed during backpropagation. */
void weight_update(int index)
{
int i,j,t;
int tmp;
double muXdelta;
int numHidXnumIn = numHid*numIn;

  for(i=0;i<numOut;i++)
  {
    muXdelta = mu * deltaOut[i];
    for(j=0;j<numHid;j++)
      W2L[c2MAT(i,j,numOut)] += muXdelta * hidOut[j];
    B2L[i] += muXdelta;
  }

  for(i=0;i<numHid;i++)
  {
    muXdelta = mu * deltaHid[i];
    for(t=0;t<numTaps;t++)
    {
      tmp = index - t;
      for(j=0;j<numIn;j++)
        W1L[c2MAT_tap(i,j,t,numHid,numHidXnumIn)] += muXdelta * 
	                                         InpVects[c2MAT(j,tmp,numIn)]; 
    }
    B1L[i] += muXdelta;
  }
}

/* Calc Weight Update ---------------------------------------------------------
  Used in batch mode only. Accumulates the weight changes during the training
 epoch in order to approximate the direction of gradient decent. The weight
 changes are not used until the end of the training epoch.*/
void calcWeightUpdate(int index)
{
int i,j,t;
int tmp;
int numHidXnumIn = numHid*numIn;

  for(i=0;i<numOut;i++)
  {
    for(j=0;j<numHid;j++)
      W2change[c2MAT(i,j,numOut)]+=deltaOut[i] * hidOut[j];
    B2change[i]+=deltaOut[i];
  }
			  
  for(i=0;i<numHid;i++)
  {
    for(t=0;t<numTaps;t++)
    {
      tmp = index -t;
      for(j=0;j<numIn;j++)
        W1change[c2MAT_tap(i,j,t,numHid,numHidXnumIn)] += deltaHid[i] * 
	                                        InpVects[c2MAT(j,tmp,numIn)];
    }
    B1change[i]+=deltaHid[i];
  }
}

/* Batch Weight Update --------------------------------------------------------
  Effects the weight changes calculated by calcWeightUpdate. Called at the end
 of the training epoch when using batch mode.*/
void batchWeightUpdate(int numVects)
{
int i,j,t,mindx;
int numHidXnumIn = numHid*numIn;

  for(i=0;i<numOut;i++)
  { for(j=0;j<numHid;j++)
    { 
      mindx=c2MAT(i,j,numOut);
      W2L[mindx]+=W2change[mindx] * mu;
      W2change[mindx]=0.0;
    }
    B2L[i]+=B2change[i] * mu;
    B2change[i]=0.0;
  }

  for(i=0;i<numHid;i++)
  { for(t=0;t<numTaps;t++)
    { for(j=0;j<numIn;j++)
      {
        mindx=c2MAT_tap(i,j,t,numHid,numHidXnumIn); 
        W1L[mindx]+= W1change[mindx] * mu;
	W1change[mindx]= 0.0;
      }
    }
    B1L[i]+=B1change[i] * mu;
    B1change[i] = 0.0;
  }
}

/* Calc MSE -------------------------------------------------------------------
   Calculates the approximate Mean Squared Error of each network output over
 the current epoch by dividing the cumulative squared error for that output 
 (accumulated over the epoch) by the number of input patterns in the epoch.
   The MSE is stored in an array, and also displayed to the standard output
 if the epoch number is an integer multiple of the display period. */
void calc_MSE(int epoch)
{
int i,j;

  for(i=0;i<numOut;i++)		/* compute MSE for each output */
  {
    MSE[c2MAT(i,epoch,numOut)]= cumSqErr[i] /(double)(numInpVects - numTaps+1);
    cumSqErr[i]=0.0;
  }
  if (((epoch+1) % displayPeriod)==0) /* IF it's time to display MSE */
  {
    printf("Epoch: %d",epoch+1);	/*   then display it! */
    for(i=0;i<numOut;i++)
      printf("\t%f",MSE[c2MAT(i,epoch,numOut)]);
    printf("\n");
  }
}


/* Shuffle ------------------------------------------------------------------
   Used in stochastic training to re-order the training vectors after each 
 presentation epoch. The shuffling is done by exchanging the values of two
 randomly chosen indexes, repeated by the number of training vectors.
   After each call, randi contains an array of indices in a newly 
 randomized order. During training, randi[i] is used as the index into the 
 training vector matrix instead of the index i. */
void shuffle(int numVects)
{
int i;
double rndm;
int tmp,indx1,indx2;
int tmp1;

  tmp1 = numInpVects - numTaps + 1;
  for(i=0;i<numVects;i++)
  {
    rndm = drand48();
    indx1= (int) (rndm * (double) tmp1) + numTaps - 1;
    rndm = drand48();
    indx2= (int) (rndm * (double) tmp1) + numTaps - 1;

    tmp = randi[indx1];
    randi[indx1]=randi[indx2];
    randi[indx2]=tmp;
  }
}



/*----------------------------------------------------------------------*/
allocReturnMatrices(mxArray *plhs[])
{
  plhs[0] = mxCreateDoubleMatrix(numHid,(numTaps*numIn), mxREAL); /* creating W1L */
  plhs[1] = mxCreateDoubleMatrix(numHid,1, mxREAL);               /* creating B1L */
  plhs[2] = mxCreateDoubleMatrix(numOut,numHid, mxREAL);		/* creating W2L */
  plhs[3] = mxCreateDoubleMatrix(numOut,1, mxREAL);		/* creating B2L */
  plhs[4] = mxCreateDoubleMatrix(numOut,numEpochs, mxREAL);	/* creating MSE */
}

/*----------------------------------------------------------------------*/
assignMatrixPointers(mxArray *plhs[], mxArray *prhs[])
{
  W1R = mxGetPr(prhs[0]);
  B1R = mxGetPr(prhs[1]);
  W2R = mxGetPr(prhs[2]);
  B2R = mxGetPr(prhs[3]);
  InpVects = mxGetPr(prhs[4]);
  TargVects = mxGetPr(prhs[5]);

  W1L = mxGetPr(plhs[0]);
  B1L = mxGetPr(plhs[1]);
  W2L = mxGetPr(plhs[2]);
  B2L = mxGetPr(plhs[3]);
  MSE = mxGetPr(plhs[4]);
}

/*----------------------------------------------------------------------*/
allocInternalMemory(void)
{
  hidOut=(double *) mxCalloc(numHid , sizeof(double));
  if(hidOut == NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  netOut=(double *) mxCalloc(numOut , sizeof(double));
  if(netOut ==NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  deltaHid = (double *) mxCalloc (numHid , sizeof(double));
  if(deltaHid==NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  deltaOut = (double *) mxCalloc (numOut , sizeof(double));
  if(deltaOut==NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  W1change  = (double *) mxCalloc ((numHid*numTaps*numIn) , sizeof(double));
  if(W1change == NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  B1change  = (double *) mxCalloc (numHid, sizeof(double));
  if(B1change == NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  W2change = (double *) mxCalloc ((numHid*numOut) , sizeof(double));
  if(W2change == NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  B2change = (double *) mxCalloc (numOut, sizeof(double));
  if(B2change == NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  cumSqErr=(double *) mxCalloc (numOut , (sizeof(double)));
  if(cumSqErr ==NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");

  randi = (int *) mxCalloc(numInpVects , sizeof(int));
  if(randi==NULL)
    mexErrMsgTxt("Unable to Allocate Memory!\n");
}


