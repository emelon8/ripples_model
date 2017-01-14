#ChAT model ELIF
import numpy as np

def chat_neuron(v,Ib,w,b,bh,Inoise,dt,C,gL,EL,Eb,slp,Vr,Vt,taub,taubh,tauw,gBmax,a,B,Ie,sampa,sgaba,snmda):
    ### slow K+ current
    binf=0.14+0.81/(1+np.exp((v+22.46)/-8.08))
    bhinf=0.08+0.88/(1+np.exp((v+60.23)/5.69))
    bdt=((binf-b)/taub)*dt
    bhdt=((bhinf-bh)/taubh)*dt
    b=b+bdt
    bh=bh+bhdt
    Ib=b*bh*gBmax*(v-Eb)

    ##voltage
    fv_i=(-gL*(v-EL)+gL*slp*np.exp((v-Vt)/slp)-w-Ib+Ie+Inoise)/C

    ##w (spike dependent adaptation current)
    dw=((-w+a*(v-EL))/tauw)*dt

    k1v=dt*fv_i

    v[i+1]=v[i]+k1v
    w[i+1]=w[i]+dw

    if v[i]>0 and pulse_start/dt<=i and i<(pulse_start+pulse_duration)/dt:
        v[i+1]=Vr
        v[i]=50
        w[i+1]=w[i]+B
        spiketimes=np.append(spiketimes,i*dt/1000)
        Ib[i]=b*bh*gBmax*(v[i]-Eb)
    elif v[i]>0:
        v[i+1]=Vr
        v[i]=50
        w[i+1]=w[i]+B
        Ib[i]=b*bh*gBmax*(v[i]-Eb)

    return v,Ib,w,b,bh