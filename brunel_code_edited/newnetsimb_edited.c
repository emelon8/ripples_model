#include <math.h>
#include <stdio.h>

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
#define tprestim 10000 
#define dt 0.05
#define ivisual 0
/****************** Neurons ********************/
#define taumE 20.0
#define taumI 10.0
#define nuE 0.002
#define nuI 0.006
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

FILE *inp, *paramet, *raster, *intracel, *synaps, *rat, *activ, *ccfile, *fopen();

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
	float IEXT, IAMPA, INMDA, IGABA;
	float IAMPArk, INMDArk, IGABArk, IEXTrk;
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
			if (ran1(&idum) < epsilon) { /*if the random number is less than the probability of connection...*/
				addr[nconn[jn]][jn] = in; /*nconn[jn] is the number of connections that neurons jn receives. each column of addr contains the numbers of the presynaptic neurons that connect to the postsynaptic neuron represented by that column*/
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
		coefEXT = jext*mcEXT;
		coefAMPA = modulAMPA*je*(1. - xNMDArec)*mcAMPA;
		coefGABA = modulGABA*ji*mcGABA;
		coefNMDA = modulNMDA*je*xNMDArec*mcNMDA;
		/****** s variables *************/
		tminx = icycle%CYCSIZE;
		for (ineur = 0; ineur < nneur; ineur++) {
			if (ineur == NE) {
				threshold = thresholdI; reset = resetI; je = gie; ji = gii; jext = giext;
				mcAMPA = mcAMPAi; mcNMDA = mcNMDAi; mcGABA = mcGABAi; mcEXT = mcEXTi;
				dtAMPArise = dtAMPAirise; dtAMPAdecay = dtAMPAidecay; dtGABAdecay = dtGABAidecay;
				tauarp = tarpI; zdec = zdecayi; tdec = taumI; dtm = dtmi; freq = nuI; nuext = nuexti;
				coefEXT = jext*mcEXT;
				coefAMPA = modulAMPA*je*(1. - xNMDArec)*mcAMPA;
				coefGABA = modulGABA*ji*mcGABA;
				coefNMDA = modulNMDA*je*xNMDArec*mcNMDA;
			}
			fspk = hrece[ineur][tminx];
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
			nspkext = poidev(cEXT*nuext*dt, &idum);
			xEXTk1 = nspkext - dtEXTrise*xEXT[ineur];
			sEXTk1 = dt*xEXT[ineur] - dtEXTdecay*sEXT[ineur];
			xEXTk2 = nspkext - dtEXTrise*(xEXT[ineur] + 0.5*xEXTk1);
			sEXTk2 = dt*(xEXT[ineur] + 0.5*xEXTk1) - dtEXTdecay*(sEXT[ineur] + 0.5*sEXTk1);
			xEXT[ineur] += xEXTk2;

			IEXT = (VE - hi[ineur])*jext*mcEXT*sEXT[ineur];
			IAMPA = INMDA = IGABA = 0.;
			IAMPArk = INMDArk = IGABArk = 0.;
			IAMPA = modulAMPA*sAMPA[ineur] * (VE - hi[ineur])*je*(1. - xNMDArec)*mcAMPA;
			INMDA = modulNMDA*sNMDA[ineur] * (VE - hi[ineur])*je*xNMDArec*mcNMDA;
			IGABA = modulGABA*sGABA[ineur] * (VI - hi[ineur])*ji*mcGABA;
			hik1 = dtm*(IEXT + IAMPA + INMDA + IGABA - hi[ineur]);
			hirk = hi[ineur] + 0.5*hik1;
			IEXT = (VE - hirk)*jext*mcEXT*(sEXT[ineur] + 0.5*sEXTk1);
			IAMPArk = modulAMPA*sAMPArk[ineur] * (VE - hirk)*je*(1. - xNMDArec)*mcAMPA;
			INMDArk = modulNMDA*sNMDArk[ineur] * (VE - hirk)*je*xNMDArec*mcNMDA;
			IGABArk = modulGABA*sGABArk[ineur] * (VI - hirk)*ji*mcGABA;
			hik2 = dtm*(IEXT + IAMPArk + INMDArk + IGABArk - hirk);
			sEXT[ineur] += sEXTk2;
			hi[ineur] += hik2;
			/****** Is neuron refractory? *******/
			if (tspki[ineur] >= 0 && (icycle - tspki[ineur] < tauarp*sub)) {
				vi[ineur] = 0;
				hi[ineur] = reset;
			}
			/****** Is neuron above threshold? ******/
			if (hi[ineur] > threshold) vi[ineur] = 1;
			else vi[ineur] = 0;
			tminx = icycle%CYCSIZE;
			if (ineur == 0) {
				/**printf("%d %d %.2f %.2f %.2f %.2f %.2f\n",icycle/sub,ineur,IEXT,IAMPA,INMDA,IGABA,hi[ineur]);**/
				fprintf(intracel, "%.3f %.3f %.3f %.3f %.3f %.3f ", (float)(icycle) / (float)(sub), -IEXT, -IAMPA, -INMDA, -IGABA, (1. - vi[ineur])*(hi[ineur] - 70.));
			}
			if (ineur == NE + 1) fprintf(intracel, "%.3f %.3f %.3f %.3f %.3f\n", -IEXT, -IAMPA, -INMDA, -IGABA, (1. - vi[ineur])*(hi[ineur] - 70.));
			if (vi[ineur] == 1)  {
				tspki[ineur] = icycle;
				hi[ineur] = reset;
				spkno[ineur]++;
				jcurrdyn = 0;
				for (is = 0; is < nconn[ineur]; is++) {
					tminx = (icycle + syn[is][ineur]) % CYCSIZE;
					if (ineur < NE) hrece[addr[is][ineur]][tminx]++;
					else hreci[addr[is][ineur]][tminx]++;
				}
				if (ineur < NE) exact[icycle / sub]++;
				else inact[icycle / sub]++;
				/** if(ineur==NE+1) printf("Spike emitted at %d\n",tspki[ineur]); **/
			}
			tminx = icycle%CYCSIZE;
			hrece[ineur][tminx] = 0;
			hreci[ineur][tminx] = 0;
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
			printf("%d ", icycle / sub + globaltime); /*prints the millisecond as it gets to it when it loops through*/
			printf("%3d %3d\n", exact[icycle / sub], inact[icycle / sub]); /*prints the firing rate during the previous millisecond*/
		}
		if (icycle%sub == sub - 1) {
			fclose(raster); fclose(activ); fclose(intracel);
			raster = fopen(wrast, "a");
			activ = fopen(wacti, "a");
			intracel = fopen(wintr, "a");
			fprintf(activ, "%d ", icycle / sub + globaltime);
			fprintf(activ, "%3d %3d\n", exact[icycle / sub], inact[icycle / sub]);
			aspkE += exact[icycle / sub];
			aspkI += inact[icycle / sub];
		}
		for (ipr = 0; ipr < NE + NI; ipr++) {
			if (vi[ipr] == 1) {
				fprintf(raster, "%.2f %d\n", (float)(icycle) / (float)(sub)+globaltime, ipr);
			}
		}
	}
	subcyc = 1000. / (float)(tprestim);
	norm = (float)(tprestim);
	printf("Prestim\t%dms\t", tprestim);
	fprintf(paramet, "Prestim\t%dms\t", tprestim);
	if (flni > 0.) aspkI *= subcyc / flni;
	if (fln > 0.) aspkE *= subcyc / fln;
	printf("\tE rate %.2f I rate %.2f\n", aspkE, aspkI);
	fprintf(paramet, "\tE rate %.2f I rate %.2f\n", aspkE, aspkI);
}


main()
{
	long idum;

	paramet = fopen("dparaTEST", "w");
	inp = fopen("fileshort", "r");
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
	gee = 0.; /*** 0.01 ***/ /*0 when no E-E connections?*/
	gie = 0.02;
	gei = 0.1;  /*** 0.1 ***/
	gii = 0.2;
	geext = 0.01; giext = 0.02;

	modulGABA = 1.;
	modulAMPA = 1.;
	modulNMDA = 1.;
	fprintf(paramet, "\nNetwork... NE=%d NI=%d cE=%d cI=%d \n", NE, NI, cE, cI);
	fprintf(paramet, "tprestim=%dms\n", tprestim);
	fprintf(paramet, "Neurons... Taum=%.0fms %.0fms Thresholds=%.0fmV %.0fmV \n\tReset=%.0fmV %.0fmV ARP=%.0fms %.0fms\n", taumE, taumI, thresholdE, thresholdI, resetE, resetI, tarpE, tarpI);
	fprintf(paramet, "Synapses... VE=%.0fmV VI=%.0fmV xNMDArec=%.2f \n\tgee=%.3f gie=%.3f gei=%.3f gii=%.3f \n\tNMDA=%.1fms, %.1fms AMPAe=%.1fms, %.1fms AMPAi=%.1fms, %.1fms \n\tGABAe=%.1fms, %.1fms \tGABAi=%.1fms, %.1fms \n\tdelay min=%.2f %.2fms max=%.2f %.2fms\n\n", VE, VI, xNMDArec, gee, gie, gei, gii, tauNMDArise, tauNMDAdecay, tauAMPAerise, tauAMPAedecay, tauAMPAirise, tauAMPAidecay, tauGABArise, tauGABAedecay, tauGABArise, tauGABAidecay, delayminE, delayminI, delaymaxE, delaymaxI);

	idum = -171;
	nuexte = 0.03;
	nuexti = 0.0275;

	/*** Fig.1: nuext=0.015  ***/
	/*** Fig.5: nuext=0.0075; nuext=(0.03, 0.0275) ***/
	/*** Fig.6: nuext=0.005 ***/
	/*** Fig.7: nuext=0.00325 ***/

	printf("nuext=%.2fHz %.2fHz\n", 1000.*nuexte, 1000.*nuexti);
	fprintf(paramet, "nuext=%.2fHz %.2fHz\n", 1000.*nuexte, 1000.*nuexti);
	fscanf(inp, "%s %s %s %s", wrast, wacti, wintr, wcc);
	raster = fopen(wrast, "w");
	activ = fopen(wacti, "w");
	intracel = fopen(wintr, "w");
	run_sim(idum);
	fclose(raster); fclose(activ); fclose(intracel);
	fclose(paramet);
}
