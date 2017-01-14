#include <math.h>
#include <stdio.h>
//#include "ran1.c"
//#include "poidev.c"
//#include "gammln.c"
//#include "gasdev.c"

/****************** Network ********************/
#define NE 4000  /* No excit neurons */
#define NI 1000  /* No Inhib neurons */
#define epsilon 0.2 /* Connection probability */
#define cE (int)(epsilon*NE)  
#define cEXT cE
#define cI (int)(epsilon*NI) 
#define CNN 1300
#define CNE 10
#define CYCSIZE 100
/****************** Simulation *****************/
#define tprestim 10000 //10000
#define dt 0.05
#define ivisual 0	
/****************** Neurons ********************/
#define taumE 20.0 //20.0 used the same values as in Wang 1999, i.e.gL = 0.025 microS, Cm = 0.5nF for pyr cells, gL = 0.02 microS Cm = 0.2nF for interneurons
#define taumI 10.0
#define nuE 0.002 // assigned to "freq" but then "freq" is not used
#define nuI 0.006 // assigned to "freq" but then "freq" is not used
#define thresholdE 18.
#define thresholdI 18.
#define resetE 11.
#define resetI 11.
#define tarpE 2.
#define tarpI 1.
/****************** Synapses ********************/
#define VE 70.0
#define VI 0.0
#define xNMDArec 0.25
#define tauNMDArise 2.
#define tauNMDAdecay 100.
#define tauEXTrise 0.4
#define NDELAY 100

FILE *volt1, *volt501, *volt1001, *volt1501, *volt2001, *volt2501, *volt3001, *volt3501, *volt4001, *volt4501, *rateei, *rateneurons, *nM1, *nM4501, *s1;

float gee, gei, gie, gii, geext, giext; /* Strengths of couplings */
float extme, extmi, extse, extsi; /* external currents per tau: means, SD, for E, I determined by MF */
float mue, mui;			/* mean-field total mean currents E, I */
float se, si;			/* mean field total SD currents */
float xe, xi;			/* fraction of local synapses */
float aspkE, aspkI; 	/* averages activities */
float fln, flni; 	/* float # E,I neurons */
int nde; 		/* number of possible delays in basic steps +1 */
float fde;		/* float(fde) */
int vi[NE + NI]; 	/* current firing */
int spkno[NE + NI]; 	/* number of spikes */
float hi[NE + NI]; 	/* current potentials */

float xAMPA[NE + NI];
float sAMPA[NE + NI];
float xEXT[NE + NI];
float sEXT[NE + NI];
float xNMDA[NE + NI];
float sNMDA[NE + NI];
float xGABA[NE + NI];
float sGABA[NE + NI];
float sAMPAk1[NE + NI];
float sNMDAk1[NE + NI];
float sGABAk1[NE + NI];
float sAMPArk[NE + NI];
float sNMDArk[NE + NI];
float sGABArk[NE + NI];
float sAMPAk2[NE + NI];
float sNMDAk2[NE + NI];
float sGABAk2[NE + NI];
float nMinf, nMtau, dnMdt, nMinfk1, nMtauk1, dnMdtk1, nMinfrk, nMtaurk, dnMdtrk;
float nM[NE + NI];
float nMk1, nMrk, nMk2;
int addr[CNN][NE + NI];/* Adresses neurones postsynaptiques */
int nconn[NE + NI];	/* Nbre connections */
char syn[CNN][NE + NI]; /* Synaptic matrix (contains both efficacy AND delay) */
int hrece[NE + NI][CYCSIZE];
int hreci[NE + NI][CYCSIZE];
int syntemp[NE + NI];         /* temporary vector for synaptic matrix generation */

int tspki[NE + NI]; 	/* Time of last spike */
int inact[tprestim]; 	/* Number of I spikes at each ms */
int exact[tprestim]; 	/* Number of E spikes at each ms */
int ncycle; 		/* No of time steps */
int globaltime;
float taueffe, taueffi, mueffe, mueffi, sigmaeffe, sigmaeffi;
float normNMDA;
float nuexte, nuexti;
float rate, rati;
char wrast[50], wpara[50], wacti[50], wrate[50], wintr[50], wcc[50];
float tauGABAedecay, tauGABAidecay, tauGABA, tauGABAe, tauGABAi, tauAMPAe, tauAMPAi, tauNMDA, tauEXT;
float modulGABA, modulAMPA, modulNMDA;
int tsimul;
float tauGABArise, tauAMPAerise, tauAMPAedecay, tauAMPAirise, tauAMPAidecay, delayminE, delayminI, delaymaxE, delaymaxI, tauEXTdecay;
float gM, EM, MVhalf, Mk, MVmax, Msigma, MCamp, MCbase;

/******* Simulation ********/

void run_sim(long idum)
{
	float poidev(float xm, long *idum);        /** recipes routine for Poisson */
	float ran1(long *idum);       /** recipes routine for uniform deviate */
	float gasdev(long *idum);    /** recipes routine for Gaussian */
	int sub; 		/* Number of dt in 1ms */
	int delminsubE, delminsubI, delmaxsubE, delmaxsubI;     /**** delays min, max in dt */
	int nspk;
	float nspkext;
	float zdecaye, zdecayi, mce, mci, je, ji, jext;
	float subcyc, syncont;
	int i, i1, ii, ipp, tminx, tminx1, neff, imu;
	int ip, in, in1, ipr, nneur, is, jn;
	int ineur, oldncycle;
	int jspke, jspki, icycle;
	int iprob, probe;
	float crossee, nume, norm;
	int incyc, iprint;
	float zcAMPAe, zcAMPAi, zcNMDA, zcGABAe, zcGABAi, zcNMDArise, zcNMDAdecay, zcEXT;
	float mcAMPAe, mcAMPAi, mcNMDAe, mcNMDAi, mcGABAe, mcGABAi, mcEXTe, mcEXTi;
	float ctmpAMPA, ctmpNMDA, ctmpGABA;
	float zdec, mcAMPA, mcNMDA, mcGABA, tdec, avpot, mcEXT;
	float mu, threshold, sigma, reset, tauarp;
	float htemp;
	float sAMPAe, sAMPAi, sNMDAe, sNMDAi, sGABAe, sGABAi;
	float IEXT, IAMPA, INMDA, IGABA, IM;
	float IAMPArk, INMDArk, IGABArk, IEXTrk, IMrk;
	int idelay, curpresyn, curdel, kn, delmindrawn, delmaxdrawn;
	float hik1, hik2, sEXTk1, sEXTk2, xNMDAk1, xNMDAk2, xAMPAk1, xAMPAk2;
	float xGABAk1, xGABAk2, xEXTk1, xEXTk2;
	float dtme, dtmi;
	float dtAMPAerise, dtGABArise, dtNMDArise, dtNMDAdecay, dtAMPAedecay, dtGABAedecay, dtGABAidecay, dtGABAdecay, dtEXTdecay, dtEXTrise;
	float dtAMPAirise, dtAMPAidecay, dtAMPArise, dtAMPAdecay, tauAMPArise, tauAMPA;
	float dtm, freq, hirk, nuext;
	int fspk;
	int jcurr[NE + NI], iexc[NE + NI], iprev[NE + NI], jcurrmax, iexcmax, jcurrdyn;
	int nconnect;
	float coefAMPA, coefGABA, coefNMDA, coefEXT;

	jcurrmax = 1300;
	sub = (int)(1. / dt + 0.01);
	printf("Sub %d dt %.3f\n", sub, dt);
	nneur = NE + NI;
	fln = (float)(NE);
	flni = (float)(NI);
	zdecaye = 1. - dt / taumE;
	zdecayi = 1. - dt / taumI;
	zcAMPAe = 1. - dt / tauAMPAe;
	zcAMPAi = 1. - dt / tauAMPAi;
	zcNMDArise = 1. - dt / tauNMDArise;
	zcNMDAdecay = 1. - dt / tauNMDAdecay;
	zcGABAe = 1. - dt / tauGABAe;
	zcGABAi = 1. - dt / tauGABAi;
	dtme = dt / taumE;
	dtmi = dt / taumI;
	dtAMPAerise = dt / tauAMPAerise;
	dtAMPAirise = dt / tauAMPAirise;
	dtAMPAedecay = dt / tauAMPAedecay;
	dtAMPAidecay = dt / tauAMPAidecay;
	dtEXTrise = dt / tauEXTrise;
	dtEXTdecay = dt / tauEXTdecay;
	dtGABArise = dt / tauGABArise;
	dtGABAedecay = dt / tauGABAedecay;
	dtGABAidecay = dt / tauGABAidecay;
	dtNMDArise = dt / tauNMDArise;
	dtNMDAdecay = dt / tauNMDAdecay;
	mcAMPAe = taumE / tauAMPAe; mcAMPAi = taumI / tauAMPAi;
	normNMDA = tauNMDAdecay*tauNMDArise;
	mcNMDAe = taumE / normNMDA; mcNMDAi = taumI / normNMDA;
	mcGABAe = taumE / tauGABAe; mcGABAi = taumI / tauGABAi;
	mcEXTe = taumE / tauEXT; mcEXTi = taumI / tauEXT;
	delminsubE = (int)(delayminE*sub + 0.01);
	delminsubI = (int)(delayminI*sub + 0.01);
	delmaxsubE = (int)(delaymaxE*sub + 0.01);
	delmaxsubI = (int)(delaymaxI*sub + 0.01);
	printf("Delay min %d %d max %d %d\n", delminsubE, delminsubI, delmaxsubE, delmaxsubI);
	printf("gee %.3f gei %.3f gie %.3f gii %.3f\n", gee, gei, gie, gii);


	/********* Initialization ***********/

	mu = mueffe; threshold = thresholdE; sigma = sigmaeffe; reset = resetE; tauAMPA = tauAMPAe; tauAMPArise = tauAMPAerise; nuext = nuexte; tauGABA = tauGABAe;
	for (in = 0; in < NE + NI; in++) for (ip = 0; ip < CYCSIZE; ip++) hrece[in][ip] = hreci[in][ip] = 0;
	for (in = 0; in < NE + NI; in++) {
		if (in == NE) {
			mu = mueffi; threshold = thresholdI; sigma = sigmaeffi; reset = resetI; tauAMPA = tauAMPAi; tauAMPArise = tauAMPAirise; nuext = nuexti; tauGABA = tauGABAi;
		}
		xEXT[in] = 0.;/*** tauEXTrise*nuext*cEXT*2.*ran1(&idum); ***/
		sEXT[in] = 0.;/*** tauEXT*nuext*cEXT*2.*ran1(&idum); ***/
		xAMPA[in] = 0.;/*** tauAMPArise*rate*cE*2.*ran1(&idum); ***/
		sAMPA[in] = 0.;/*** tauAMPA*rate*cE*2.*ran1(&idum); ***/
		xNMDA[in] = 0.;/*** tauNMDArise*rate*cE*2.*ran1(&idum); ***/
		sNMDA[in] = 0.;/*** normNMDA*cE*2.*ran1(&idum); ***/
		xGABA[in] = 0.;/*** tauGABArise*rati*cI*2.*ran1(&idum); ***/
		sGABA[in] = 0.;/*** tauGABA*rati*cI*2.*ran1(&idum); ***/
		hi[in] = reset;/*** +ran1(&idum)*(threshold-reset-1.); ***/
		sEXT[in] = 0.;
		spkno[in] = 0;
		tspki[in] = -1;
	}
	aspkE = aspkI = 0;

	printf("Fin initialisation....");

	delmindrawn = delmaxsubE;
	delmaxdrawn = delminsubE;
	for (i = 0; i < NE + NI; i++) jcurr[i] = iexc[i] = iprev[i] = 0;
	printf("Debut tirage synapses... \n");
	for (jn = 0; jn < NE + NI; jn++) {
		nconn[jn] = 0;
		for (in = 0; in < NE + NI; in++) {
			if (ran1(&idum) < epsilon) {
				addr[nconn[jn]][jn] = in;
				if (jn < NE) syn[nconn[jn]][jn] = delminsubE + (int)((delmaxsubE + 1 - delminsubE)*ran1(&idum));
				else syn[nconn[jn]][jn] = delminsubI + (int)((delmaxsubI + 1 - delminsubI)*ran1(&idum));
				nconn[jn]++;
			}
		}
	}
	for (i = 0; i < NE + NI; i++) {
		if (nconn[i] > jcurrmax) jcurrmax = jcurr[i];
	}
	printf("\njcurrmax %d %d\n", jcurrmax);


	for (i = 0; i < tsimul; i++) exact[i] = inact[i] = 0;
	printf("End of initialization... \n");

	/********* Loop on cycles ***********/

	for (icycle = 0; icycle < sub*tsimul/**&&exact[icycle/sub]<NE/3**/; icycle++) {
		/* printf("Icycle=%d\n",icycle); */
		/*** printf("cycle %d: loop on neurons\n",icycle); ***/
		/***** Loop on neurons *****/
		threshold = thresholdE; reset = resetE; je = gee; ji = gei; jext = geext;
		mcAMPA = mcAMPAe; mcNMDA = mcNMDAe; mcGABA = mcGABAe; mcEXT = mcEXTe;
		dtAMPArise = dtAMPAerise; dtAMPAdecay = dtAMPAedecay; dtGABAdecay = dtGABAedecay;
		tauarp = tarpE; zdec = zdecaye; tdec = taumE; dtm = dtme; freq = nuE; nuext = nuexte;
		coefEXT = jext*mcEXT; // 0.01 * 25 = 0.25
		coefAMPA = modulAMPA*je*(1. - xNMDArec)*mcAMPA; // 1 * 0 * 0.75 * 25 = 0
		coefGABA = modulGABA*ji*mcGABA; // 1 * 0.1 * 8 = 0.8
		coefNMDA = modulNMDA*je*xNMDArec*mcNMDA; // 1 * 0 * 0.25 * 0.1 = 0
		/****** s variables *************/
		tminx = icycle%CYCSIZE;
		for (ineur = 0; ineur < nneur; ineur++) {
			if (ineur == NE) { //sets the parameters to the parameters for inhibitory neurons from here on (for this time step)
				threshold = thresholdI; reset = resetI; je = gie; ji = gii; jext = giext;
				mcAMPA = mcAMPAi; mcNMDA = mcNMDAi; mcGABA = mcGABAi; mcEXT = mcEXTi;
				dtAMPArise = dtAMPAirise; dtAMPAdecay = dtAMPAidecay; dtGABAdecay = dtGABAidecay;
				tauarp = tarpI; zdec = zdecayi; tdec = taumI; dtm = dtmi; freq = nuI; nuext = nuexti;
				coefEXT = jext*mcEXT; // 0.02 * 12.5 = 0.25
				coefAMPA = modulAMPA*je*(1. - xNMDArec)*mcAMPA; // 1 * 0.02 * 0.75 * 12.5 = 0.1875
				coefGABA = modulGABA*ji*mcGABA; // 1 * 0.2 * 4 = 0.8
				coefNMDA = modulNMDA*je*xNMDArec*mcNMDA; // 1 * 0.02 * 0.25 * 0.05 = 0.00025
			}
			if (ineur == 1) { if (icycle == 10000) fspk = 1; else fspk = 0; }
			else { fspk = hrece[ineur][tminx]; }
			xAMPAk1 = fspk - dtAMPArise*xAMPA[ineur];
			xNMDAk1 = fspk - dtNMDArise*xNMDA[ineur];

			sAMPAk1[ineur] = dt*xAMPA[ineur] - dtAMPAdecay*sAMPA[ineur];
			sNMDAk1[ineur] = dt*xNMDA[ineur] - dtNMDAdecay*sNMDA[ineur];

			sAMPArk[ineur] = sAMPA[ineur] + 0.5*sAMPAk1[ineur];
			sNMDArk[ineur] = sNMDA[ineur] + 0.5*sNMDAk1[ineur];

			xAMPAk2 = fspk - dtAMPArise*(xAMPA[ineur] + 0.5*xAMPAk1);
			xNMDAk2 = fspk - dtNMDArise*(xNMDA[ineur] + 0.5*xNMDAk1);

			sAMPAk2[ineur] = dt*(xAMPA[ineur] + 0.5*xAMPAk1) - dtAMPAdecay*sAMPArk[ineur];
			sNMDAk2[ineur] = dt*(xNMDA[ineur] + 0.5*xNMDAk1) - dtNMDAdecay*sNMDArk[ineur];

			xAMPA[ineur] += xAMPAk2;
			xNMDA[ineur] += xNMDAk2;

			fspk = hreci[ineur][tminx];
			xGABAk1 = fspk - dtGABArise*xGABA[ineur];
			sGABAk1[ineur] = dt*xGABA[ineur] - dtGABAdecay*sGABA[ineur];
			sGABArk[ineur] = sGABA[ineur] + 0.5*sGABAk1[ineur];
			xGABAk2 = fspk - dtGABArise*(xGABA[ineur] + 0.5*xGABAk1);
			sGABAk2[ineur] = dt*(xGABA[ineur] + 0.5*xGABAk1) - dtGABAdecay*sGABArk[ineur];
			xGABA[ineur] += xGABAk2;
			/***** External inputs *******/
			nspkext = poidev(cEXT*nuext*dt, &idum); // first argument is lambda, second argument is the seed number
			xEXTk1 = nspkext - dtEXTrise*xEXT[ineur];
			sEXTk1 = dt*xEXT[ineur] - dtEXTdecay*sEXT[ineur];
			xEXTk2 = nspkext - dtEXTrise*(xEXT[ineur] + 0.5*xEXTk1);
			sEXTk2 = dt*(xEXT[ineur] + 0.5*xEXTk1) - dtEXTdecay*(sEXT[ineur] + 0.5*sEXTk1);
			xEXT[ineur] += xEXTk2;
			
			IEXT = (VE - hi[ineur])*jext*mcEXT*sEXT[ineur];
			IAMPA = INMDA = IGABA = IM = 0.;
			IAMPArk = INMDArk = IGABArk = IMrk = 0.;
			IAMPA = coefAMPA * sAMPA[ineur] * (VE - hi[ineur]);
			INMDA = coefNMDA * sNMDA[ineur] * (VE - hi[ineur]);
			IGABA = coefGABA * sGABA[ineur] * (VI - hi[ineur]);

			IM = gM * nM[ineur] * (EM - hi[ineur]);
			nMinfk1 = 1 / (1 + exp((MVhalf - hi[ineur]) / Mk));
			nMtauk1 = MCbase + MCamp*exp(-pow(MVmax - hi[ineur], 2.0) / pow(Msigma, 2));
			dnMdtk1 = (nMinfk1 - nM[ineur]) / nMtauk1;
			nMk1 = dnMdtk1*dt;

			hik1 = dtm*(IEXT + IAMPA + INMDA + IGABA + IM - hi[ineur]);
			hirk = hi[ineur] + 0.5*hik1;
			IEXT = (VE - hirk)*jext*mcEXT*(sEXT[ineur] + 0.5*sEXTk1);
			IAMPArk = coefAMPA * sAMPArk[ineur] * (VE - hirk);
			INMDArk = coefNMDA * sNMDArk[ineur] * (VE - hirk);
			IGABArk = coefGABA * sGABArk[ineur] * (VI - hirk);

			nMrk = nM[ineur] + 0.5*nMk1;
			nMinfrk = 1 / (1 + exp((MVhalf - hirk) / Mk));
			nMtaurk = MCbase + MCamp*exp(-pow(MVmax - hirk, 2.0) / pow(Msigma, 2));
			dnMdtrk = (nMinfrk - nMrk) / nMtaurk;
			nMk2 = dnMdtrk*dt;
			IMrk = gM * nMk2 * (EM - hirk);

			hik2 = dtm*(IEXT + IAMPArk + INMDArk + IGABArk + IMrk - hirk);
			sEXT[ineur] += sEXTk2;
			hi[ineur] += hik2;
			nM[ineur] += nMk2;
			/****** Is neuron refractory? *******/
			if (tspki[ineur] >= 0 && (icycle - tspki[ineur] < tauarp*sub)) {
				vi[ineur] = 0;
				hi[ineur] = reset;
			}
			/****** Is neuron above threshold? ******/
			if (hi[ineur] > threshold) vi[ineur] = 1;
			else vi[ineur] = 0;
			tminx = icycle%CYCSIZE;

			if (vi[ineur] == 1)  {
				tspki[ineur] = icycle; // time of the last spike for this neuron, used for refractory-ness
				hi[ineur] = reset; // set the suprathreshold voltage to reset
				spkno[ineur]++; // increment number of spikes for this neuron
				jcurrdyn = 0; // not used anywhere
				for (is = 0; is < nconn[ineur]; is++) { //nconn[ineur] is the number of neurons that ineur connects to
					tminx = (icycle + syn[is][ineur]) % CYCSIZE;
					if (ineur < NE) hrece[addr[is][ineur]][tminx]++;
					else hreci[addr[is][ineur]][tminx]++;
				}
				if (ineur < NE) exact[icycle / sub]++; // since icycle and sub are (int), then the result of icycle/sub is also an int and truncates any decimal remainder
				else inact[icycle / sub]++;
				/** if(ineur==NE+1) printf("Spike emitted at %d\n",tspki[ineur]); **/
			}
			tminx = icycle%CYCSIZE;
			hrece[ineur][tminx] = 0;
			hreci[ineur][tminx] = 0;
			//fprintf(volt, "%d,%f,%f\n", ineur, (float)icycle / (float)sub, hi[1]); //(float)(icycle) / (float)(sub)+globaltime
			//if (ineur >= 0 && ineur < 500) {
			//	// fprintf(volt1, "%d,%f,%f\n", ineur, (float)icycle / (float)sub, hi[ineur]); //(float)(icycle) / (float)(sub)+globaltime
			//	fprintf(volt1, "%f,\n", hi[ineur]);
			//	fprintf(nM1, "%f,\n", nM[ineur]);
			//}
			if (ineur == 1) {
				fprintf(s1, "%d,%f\n", fspk, sAMPA[ineur]);
			}
			/*if (ineur >= 500 && ineur < 1000) {
				fprintf(volt501, "%f,\n", hi[ineur]);
			}
			if (ineur >= 1000 && ineur < 1500) {
				fprintf(volt1001, "%f,\n", hi[ineur]);
			}
			if (ineur >= 1500 && ineur < 2000) {
				fprintf(volt1501, "%f,\n", hi[ineur]);
			}
			if (ineur >= 2000 && ineur < 2500) {
				fprintf(volt2001, "%f,\n", hi[ineur]);
			}
			if (ineur >= 2500 && ineur < 3000) {
				fprintf(volt2501, "%f,\n", hi[ineur]);
			}
			if (ineur >= 3000 && ineur < 3500) {
				fprintf(volt3001, "%f,\n", hi[ineur]);
			}
			if (ineur >= 3500 && ineur < 4000) {
				fprintf(volt3501, "%f,\n", hi[ineur]);
			}
			if (ineur >= 4000 && ineur < 4500) {
				fprintf(volt4001, "%f,\n", hi[ineur]);
			}*/
			/*if (ineur >= 4500 && ineur < 5000) {
				fprintf(volt4501, "%f,\n", hi[ineur]);
				fprintf(nM4501, "%f,\n", nM[ineur]);
			}*/
		}
		tminx = icycle%CYCSIZE;
		for (ineur = 0; ineur < NE + NI; ineur++) {
			sAMPA[ineur] += sAMPAk2[ineur];
			sNMDA[ineur] += sNMDAk2[ineur];
			sGABA[ineur] += sGABAk2[ineur];
		}
		/*** printf("icycle %d : end of neuron loop\n",icycle); ***/
		/******* End of loop for neurons *****/
		if (ivisual == 1 && icycle%sub == sub - 1) {
			printf("%d ", icycle / sub + globaltime);
			printf("%3d %3d\n", exact[icycle / sub], inact[icycle / sub]);
		}
		if (icycle%sub == sub - 1) {
			aspkE += exact[icycle / sub];
			aspkI += inact[icycle / sub];
		}
	}
	subcyc = 1000. / (float)(tprestim);
	norm = (float)(tprestim);
	printf("Prestim\t%dms\t", tprestim);

	if (flni > 0.) aspkI *= subcyc / flni;
	if (fln > 0.) aspkE *= subcyc / fln;
	printf("\tE rate %.2f I rate %.2f\n", aspkE, aspkI);
}


int main()
{
	long idum;

	globaltime = 0;
	tsimul = tprestim;
	tauGABAedecay = 5.;
	tauGABAidecay = 5.;
	tauGABArise = 0.5;
	tauGABAe = tauGABAedecay*tauGABArise;
	tauGABAi = tauGABAidecay*tauGABArise;
	tauAMPAerise = 0.4;
	tauAMPAedecay = 2.;
	tauAMPAe = tauAMPAerise*tauAMPAedecay;
	tauAMPAirise = 0.4;
	tauAMPAidecay = 2.;
	tauAMPAi = tauAMPAirise*tauAMPAidecay;
	tauNMDA = tauNMDArise*tauNMDAdecay;
	tauEXTdecay = 2.;
	tauEXT = tauEXTrise*tauEXTdecay;
	delayminI = 0.5;
	delaymaxI = 0.5;
	delayminE = 1.;
	delaymaxE = 1.;

	/*** Fig.1: GABA=(1,0.5,5) ***/
	/*** Fig.5: GABA=(0.5,0.5,5) AMPA = (1,0.4,2) ***/
	/*** Fig.6: GABA=(0.5,0.5,5) AMPAE = (0.5,0.4,2) AMPAI = (0.5,0.2,1) ***/
	/*** Fig.7: GABA=(1.5,1.5,8) AMPAE = (1.5,0.4,2) AMPAI = (1.5,0.2,1) ***/

	rate = 0.001; rati = 0.015;
	gee = 0.; /*** 0.01 ***/
	gie = 0.02;
	gei = 0.1;  /*** 0.1 ***/
	gii = 0.2;
	geext = 0.01; giext = 0.02; //geext = 0.01; giext = 0.02
	gM = 2; // taken from Ermentrout and Terman, 2010, p. 85 (which took it from Destexhe and Pare, 1999?).
	EM = -77; // all these taken from Izhikevich, 2007, p. 45, 47 (which took it from Robbins et al., 1992).
	MVhalf = -44;
	Mk = 8;
	MVmax = -50;
	Msigma = 25;
	MCamp = 320;
	MCbase = 20;

	modulGABA = 1.;
	modulAMPA = 1.;
	modulNMDA = 1.;

	idum = -171; // -171
	nuexte = 0.03; //0.03 used to calculate the Poisson process rate for excitatory neurons = 24000 Hz / 800 external synapses per neuron / 1000 ms in a second
	nuexti = 0.0275; //0.0275 used to calculate the Poisson process rate for inhibitory neurons = 22000 Hz / 800 external synapses per neuron / 1000 ms in a second

	/*** Fig.1: nuext=0.015  ***/
	/*** Fig.5: nuext=0.0075; nuext=(0.03, 0.0275) ***/
	/*** Fig.6: nuext=0.005 ***/
	/*** Fig.7: nuext=0.00325 ***/

	printf("nuext=%.2fHz %.2fHz\n", 1000.*nuexte, 1000.*nuexti);
	
	/*fopen_s(&volt501, "10000ms_voltage_501-1000_voltages.txt", "a");
	fopen_s(&volt1001, "10000ms_voltage_1001-1500_voltages.txt", "a");
	fopen_s(&volt1501, "10000ms_voltage_1501-2000_voltages.txt", "a");
	fopen_s(&volt2001, "10000ms_voltage_2001-2500_voltages.txt", "a");
	fopen_s(&volt2501, "10000ms_voltage_2501-3000_voltages.txt", "a");
	fopen_s(&volt3001, "10000ms_voltage_3001-3500_voltages.txt", "a");
	fopen_s(&volt3501, "10000ms_voltage_3501-4000_voltages.txt", "a");
	fopen_s(&volt4001, "10000ms_voltage_4001-4500_voltages.txt", "a");*/
	fopen_s(&s1, "10000ms_s_1.txt", "a");
	run_sim(idum);
	fclose(s1); //fclose(volt501); fclose(volt1001); fclose(volt1501); fclose(volt2001); fclose(volt2501); fclose(volt3001); fclose(volt3501); fclose(volt4001);
	
	return 0;
}