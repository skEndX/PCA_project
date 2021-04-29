S=[];   %이미지 넣을 기본 배열
M=25;   %25개
icol=92;    %가로
irow=112;   %세로
N=icol*irow;

for i=1:3
    rgb=imread(['C:\Users\김다희\Documents\MATLAB\database\training\' num2str(i,'%03g') '.bmp']);
    img=rgb2gray(rgb);
    temp=reshape(img',N,1); %사진 하나를 칼럼벡터로 만듦
    temp=double(temp);  %정확하게 계산하기 위해
    S=[S temp]; %25개의 칼럼벡터 nxm을 만듦  rgb2gray(RGB);
end
for i=4:M
    img=imread(['C:\Users\김다희\Documents\MATLAB\database\training\' num2str(i,'%03g') '.bmp']);
    temp=reshape(img',N,1); %사진 하나를 칼럼벡터로 만듦
    temp=double(temp);  %정확하게 계산하기 위해
    S=[S temp]; %25개의 칼럼벡터 nxm을 만듦rgb2gray(RGB);
end

%정규화
X=zeros(N,M);
for i=1:M
    temp=S(:,i);
    m=mean(temp);   %평균
    st=std(temp);   %표준편차
    z=(temp-m)/st;
    X(:,i)=z*128+128; %오차가 나는 것을 방지하기 위해 평균 표준편차가 128로 지정(정규화 과정)
end

%정규화 이미지 출력
figure(2);
for i=1:M
    img=reshape(X(:,i), icol, irow);    %출력하기 위해 다시 복원
    img=uint8(img');
    subplot(ceil(sqrt(M)), ceil(sqrt(M)), i);
    imshow(img);
    pause(0.01);
end

%평균 얼굴 계산
m=mean(X,2); %(X,2) row별로 평균 계산
img=reshape(m, icol, irow); %복원
img=uint8(img');
figure(3);
imshow(img);
title('평균 얼굴', 'fontsize', 10);

A=zeros(N,M);   %각 얼굴별로 평균을 뺀 얼굴을 저장
for i=1:M
    A(:,i)=double(X(:,i))-m;
end



%공분산 행렬 계산
L=(A'*A)/(M-1); %25x25
[V, lamda]=eig(L);  %왼쪽 벡터 오른쪽 value, lamda가 오름차순으로 정렬되 있음
%index는 eigenvalue가 원래 있던 row위치를 알려줌
[lamda, index]=sort(lamda, 'descend');  %내림차순으로 lamda sorting value를 맨위에 나오게 함
Vtemp=zeros(size(V));
len=length(index); %25
for i=1:len
    Vtemp(:,i)=V(:,len+1-index(i)); %컬럼의 모든 로우, 1부터 시작이니까 +1를 더해주고, index(i)는 1,2,3,4...이렇게 나옴
    %Vtemp는 eigenvalue가 큰 것부터 컬럼끼리 정리됨
end
V=Vtemp;    %다시 저장

U=[];   %공분산 v는 l에 관한 벡터, av는 c에 관한 벡터
for i=1:size(V,2)   %U size 10304x25, 25개의 얼굴의 공분산을 U에 저장
    U=[U (A*V(:,i))];   %c에 관한 eigenvector들이 들어있는 행렬
end

U_norm=[];%U에 저장한 값들 정규화한걸 다시 저장하는 과정
for i=1:size(U,2)
    U_norm(:,i)=U(:,i)./norm(U(:,i));
end

%평균얼굴을 빼고 그것들의 공분산을 구해서 고유얼굴들을 구함
figure(4);
for i=1:size(U_norm,2)
    img=reshape(U_norm(:,i), icol, irow);
    img=img';
    img=histeq(img,255);
    subplot(ceil(sqrt(M)), ceil(sqrt(M)),i);
    imshow(img);
end

% clear all; close all;
% p=imread('001.bmp'); ph=histeq(p);
% figure(1); clf; subplot(1,2,1); imshow(p); title('original image','fontsize',15);
% subplot(1,2,2); imhist(p); title('original histogram','fontsize',15);
%
% figure(2); clf; subplot(1,2,1); imshow(ph); title('equalized image','fontsize',15);
% subplot(1,2,2); imhist(ph); title('equalized hisrogram','fontsize',15);

%가중치 벡터를 만듦
Omega=[];
for h=1:size(A,2)%25까지 반복
    WW=[];
    for i=1:size(U_norm,2)%25까지 반복
        WeightOfImage=dot(A(:,h), U_norm(:,i));%내적하는거, 스칼라 값들이 나옴
        WW=[WW;  WeightOfImage];%한 얼굴에 대한 가중치벡터를 저장
    end
    Omega=[Omega WW];%25개의 얼굴들의 가중치벡터 행렬
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%새로들어온 사진 똑같은 과정으로 인식
InputImage=input('인식할 얼굴의 번호를 입력하세요\n');
if InputImage>0 && InputImage<4
    rgb=imread(['C:\Users\김다희\Documents\MATLAB\database\test\' num2str(InputImage, '%03g') '.bmp']);
    InputImage=rgb2gray(rgb);
else
    InputImage=imread(['C:\Users\김다희\Documents\MATLAB\database\test\' num2str(InputImage, '%03g') '.bmp']);
end
temp=reshape(double(InputImage)', irow*icol, 1);
me=mean(temp);
st=std(temp);
Z=(temp-me)/st;
x_hat=Z*128+128;
a_hat=x_hat-m;

Omega_h=[];     %새로운 이미지의 가중치 벡터 계산
for i=1:M
    O=dot(a_hat,U_norm(:,i));
    Omega_h=[Omega_h; O];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%거리 계산
e=[];
for i=1:size(Omega,2)
    mag=norm(Omega_h-Omega(:,i));%새로들어온 가중치벡터에서 학습시킨 가중치벡터를 뺌
    e=[e mag];  %벡터합침 25x25
end
figure()    %거리들을 그래프로 그림
kk=1:size(e,2);
stem(kk,e)
xlim([0 26]);
ax=gca;
ax.XTick=1:1:25;
xlabel('학습이미지', 'Fontsize', 15);
ylabel('가중치벡터 간의 거리', 'Fontsize', 15);


[temp idx]=min(e);%거리가 제일 짧은 걸 저장, 가중치 tmep 몇번째 있었다도 저장 index
if temp<7000    
    figure(5);
    subplot(1,2,1)
    imshow(InputImage); title('입력이미지', 'Fontsize', 15);
    subplot(1,2,2)
    imshow(uint8(reshape(S(:,idx),icol, irow)'));
    title('인식된 학습이미지', 'Fontsize',15);
    disp(['이 얼굴은 ' num2str(idx), '번 얼굴입니다!'])
else %사진이 없을 경우(가중치가 가장 작은 temp가 7000을 안넘을 경우)
    figure(5);
    imshow(InputImage); title('입력이미지', 'Fontsize', 15);
    disp('인식된 이미지가 없습니다.');
end
