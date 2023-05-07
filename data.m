sjsl=12098;
zhengdian=zeros(1,sjsl);
fudian=zeros(1,sjsl);
qiangqin=zeros(1,sjsl);
qiangshu=zeros(1,sjsl);
ruoqin=zeros(1,sjsl);
jiC=zeros(1,sjsl);
jiG=zeros(1,sjsl);
jiP=zeros(1,sjsl);
L=0;
for i=1:sjsl
    l=length(zifu(1,:));
    if l>L
        L=l;
    end
end
dian=zeros(sjsl,L);
shui=zeros(sjsl,L);
dianxin=zeros(sjsl,L);
for i=1:sjsl
    zifu=char(mn(i));
    L=length(zifu(1,:));
    for j=1:L
        if zifu(1,j)=='H'||zifu(1,j)=='K'||zifu(1,j)=='R'
            dian(i,j)=1;
            zhengdian(i)=zhengdian(i)+1;
        end
        if zifu(1,j)=='D'||zifu(1,j)=='E'
            dian(i,j)=-1;
            fudian(i)=fudian(i)+1;
        end
        if zifu(1,j)=='D'||zifu(1,j)=='E'||zifu(1,j)=='R'||zifu(1,j)=='N'||zifu(1,j)=='Q'||zifu(1,j)=='K'||zifu(1,j)=='H'
            shui(i,j)=1;
            qiangqin(i)=qiangqin(i)+1;
        end
        if zifu(1,j)=='S'||zifu(1,j)=='T'||zifu(1,j)=='W'||zifu(1,j)=='Y'
            shui(i,j)=2;
            qiangshu(i)=qiangshu(i)+1;
        end
        if zifu(1,j)=='A'||zifu(1,j)=='F'||zifu(1,j)=='I'||zifu(1,j)=='L'||zifu(1,j)=='M'||zifu(1,j)=='V'
            shui(i,j)=3;
            ruoqin(i)=ruoqin(i)+1;
        end
        if zifu(1,j)=='C'
            shui(i,j)=4;
            jiC(i)=jiC(i)+1;
        end
        if zifu(1,j)=='G'
            shui(i,j)=5;
            jiG(i)=jiG(i)+1;
        end
        if zifu(1,j)=='P'
            shui(i,j)=6;
            jiP(j)=jiP(j)+1;
        end
    end
    pzhengdian(i)=zhengdian(i)/L;
    pfudian(i)=fudian(i)/L;
    pwudian(i)=(L-zhengdian(i)-fudian(i))/L;
    pqiangqin(i)=qiangqin(i)/L;
    pqiangshu(i)=qiangshu(i)/L;
    pruoqin(i)=ruoqin(i)/L;
    pC(i)=jiC(i)/L;
    pG(i)=jiG(i)/L;
    pP(i)=jiP(i)/L;
    Hdian(i)=0;
    if pzhengdian(i)~=0
        Hdian(i)=Hdian(i)+pzhengdian(i)*log2(pzhengdian(i));
    end
    if pfudian(i)~=0
        Hdian(i)=Hdian(i)+pfudian(i)*log2(pfudian(i));
    end
    if pwudian(i)~=0
        Hdian(i)=Hdian(i)+pwudian(i)*log2(pwudian(i));
    end
    Hdian(i)=-Hdian(i);
    Hshui(i)=0;
    if pqiangqin(i)~=0
        Hshui(i)=Hshui(i)+pqiangqin(i)*log2(pqiangqin(i));
    end
    if pqiangshu(i)~=0
        Hshui(i)=Hshui(i)+pqiangshu(i)*log2(pqiangshu(i));
    end
    if pruoqin(i)~=0
        Hshui(i)=Hshui(i)+pruoqin(i)*log2(pruoqin(i));
    end
    if pC(i)~=0
        Hshui(i)=Hshui(i)+pC(i)*log2(pC(i));
    end
    if pG(i)~=0
        Hshui(i)=Hshui(i)+pG(i)*log2(pG(i));
    end
    if pP(i)~=0
        Hshui(i)=Hshui(i)+pP(i)*log2(pP(i));
    end
    Hshui(i)=-Hshui(i);
end