

clear all;
tic
if true
 % row_raw=832;  col_raw=480;
 % row_raw=2560;  col_raw=1600;
   row_raw=1920;  col_raw=1080;
%row_raw=416;  col_raw=240;
%row_raw=1280;  col_raw=720;
%row_raw=1024;  col_raw=768;
%row_raw=1024;  col_raw=1024;
fin=fopen('Cactus_1920x1080_50.raw','r');
I=fread(fin,row_raw*col_raw,'uint8=>uint8'); 
Z=reshape(I,row_raw,col_raw);
Z=Z';
%k=imshow(Z);
%im=imshow(Z);
end
%Ximage=imcrop(Z,[500 500 127 127]);%GBM2/nci20
%Ximage=imcrop(Z,[500 500 63 63]);%GBM2/nci20 small
%Ximage=imcrop(Z,[400 250 63 63]);%GBM2/nci20 small
%Ximage=imcrop(Z,[670 380 127 127]);%scmap1
%Ximage=imcrop(Z,[670 380 63 63]);%scmap1 small
%Ximage=imcrop(Z,[425 300 127 127]);%scmap2
%Ximage=imcrop(Z,[425 300 63 63]);%scmap2 small
%Ximage=imcrop(Z,[538 324 63 63]);%sc-web browsing small
%Ximage=imcrop(Z,[400 432 127 127]);%scrobot1
%Ximage=imcrop(Z,[662 324 127 127]);%scrobot2
%Ximage=imcrop(Z,[600 200 127 127]); % for traffic1
%Ximage=imcrop(Z,[600 200 63 63]); % for traffic1 small

%Ximage=imcrop(Z,[679 319 63 63]); % for traffic1 small
%Ximage=imcrop(Z,[404 784 63 63]); % for traffic2 small
%Ximage=imcrop(Z,[1050 600 127 127]); % for people on street1
%Ximage=imcrop(Z,[2090 325 127 127]); % for people on street2
%Ximage=imcrop(Z,[1050 600 63 63]); % for people on street SMALL
%Ximage=imcrop(Z,[2090 325 63 63]); % for people on street2 small
%Ximage=imcrop(Z,[1050 600 255 255]); % for people on street large
%Ximage=imcrop(Z,[1000 1000 127 127]); %for traffic2
%Ximage=imcrop(Z,[220 150 63 63]); %basketballdrilltext1
%Ximage=imcrop(Z,[200 100 127 127]); %basketballdrilltext1
%Ximage=imcrop(Z,[630 20 127 127]);%basketballdrilltext2
%Ximage=imcrop(Z,[450 300 127 127]);%chinaspeed1
%Ximage=imcrop(Z,[450 300 63 63]);%chinaspeed1 small
%Ximage=imcrop(Z,[328 435 127 127]);%chinaspeed2
%Ximage=imcrop(Z,[328 435 63 63]);%chinaspeed2 small
%Ximage=imcrop(Z,[740 640 127 127]);%chinaspeed3
%Ximage=imcrop(Z,[150 150 127 127]);%blowing bubble1
%Ximage=imcrop(Z,[150 150 63 63]);%blowing bubble1 small
%Ximage=imcrop(Z,[46 170 127 127]);%blowing bubble2
%Ximage=imcrop(Z,[46 170 63 63]);%blowing bubble2 small
%Ximage=imcrop(Z,[1150 135 127 127]); %for cactus1
%Ximage=imcrop(Z,[1150 135 63 63]); %for cactus1 SMALL
%Ximage=imcrop(Z,[221 184 127 127]); %for cactus2
Ximage=imcrop(Z,[221 184 63 63]); %for cactus2 small
%Ximage=imcrop(Z,[150 520 127 127]); %for cactus3
%Ximage=imcrop(Z,[150 520 63 63]); %for cactus3
%Ximage=imcrop(Z,[600 400 127 127]); %for kimono
%Ximage=imcrop(Z,[825 344 127 127]); %for kimono2
%Ximage=imcrop(Z,[419 724 127 127]);%for kimono3
%Ximage=imcrop(Z,[419 724 31 31]);%for kimono3 small
%Ximage=imcrop(Z,[419 724 15 15]);%for kimono3 very small
%Ximage=imcrop(Z,[600 400 63 63]); %for kimono small
%Ximage=imcrop(Z,[825 344 63 63]); %for kimono2 small
%Ximage=imcrop(Z,[625 110 127 127]);%slide editing
%Ximage=imcrop(Z,[550 300 127 127]);%slide show
%Ximage=imcrop(Z,[550 300 63 63]);%slide show small
%Ximage=imcrop(Z,[820 300 127 127]);%KristenAndSara1
%Ximage=imcrop(Z,[820 300 63 63]);%KristenAndSara1 small
%Ximage=imcrop(Z,[875 290 63 63]);%KristenAndSara3 small
%Ximage=imcrop(Z,[400 514 127 127]);%KristenAndSara2
%Ximage=imcrop(Z,[400 514 63 63]);%KristenAndSara2 small
%%%%%%%%%%%%%%%%%%%%%%%Ximage=imcrop(Z,[500 210 31 31]);%Johnny
%Ximage=imcrop(Z,[688 190 63 63]);%Johnny small
%Ximage=imcrop(Z,[700 200 127 127]);%Johnny 
%Ximage=imcrop(Z,[737 40 63 63]);%Johnny2 small
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Ximage=imcrop(Z,[737 40 127 127]);%Johnny2 small
%Ximage=imcrop(Z,[661 441 127 127]);%fourpeople1
%Ximage=imcrop(Z,[661 441 63 63]);%fourpeople1 small
%Ximage=imcrop(Z,[700 500 63 63]);%fourpeople3 small
%Ximage=imcrop(Z,[340 470 63 63]);%fourpeople4 small
%Ximage=imcrop(Z,[382 270 127 127]);%fourpeople2
%Ximage=imcrop(Z,[382 270 63 63]);%fourpeople2 small
%Ximage=imcrop(Z,[280 80 127 127]);%race horse class D
%  Ximage=imcrop(Z,[280 80 63 63]);%race horse class D small
  
% Ximage=imcrop(Z,[570 230 127 127]);%party scene1
%Ximage=imcrop(Z,[570 230 31 31]);%party scene1
 %Ximage=imcrop(Z,[570 230 63 63]);%party scene1 small
% Ximage=imcrop(Z,[670 304 127 127]);%party scene2
%Ximage=imcrop(Z,[670 304 63 63]);%party scene2 small
%Ximage=imcrop(Z,[550 370 63 63]);%party scene3 small
 %Ximage=imcrop(Z,[500 260 127 127]);%BQ mall1
 %Ximage=imcrop(Z,[500 260 63 63]);%BQ mall1 SMALL
 % Ximage=imcrop(Z,[185 199 127 127]); %TEST sequence BQ mall2
 %Ximage=imcrop(Z,[185 199 63 63]); %BQ mall2 small
 %Ximage=imcrop(Z,[20 70 127 127]);%BQ Square1
%Ximage=imcrop(Z,[20 70 63 63]);%BQ Square1 small
%Ximage=imcrop(Z,[250 108 127 127]);%BQ Square2
%Ximage=imcrop(Z,[250 108 63 63]);%BQ Square2 small
%Ximage=imcrop(Z,[240 100 127 127]);%Basketballpass
%Ximage=imcrop(Z,[240 100 63 63]);%Basketballpass
% Ximage=imcrop(Z,[50 100 127 127]);%Basketball drill
% Ximage=imcrop(Z,[50 100 63 63]);%Basketball drill small
%%%%%%%%%%%%%%%%%%%%Ximage=imcrop(Z,[50 100 63 63]);%Basketball drill small
 %Ximage=imcrop(Z,[122 7 63 63]);%Basketball drill2 small
 %Ximage=imcrop(Z,[156 150 127 127]);%Basketball drill3 small
%Ximage=imcrop(Z,[122 7 63 63]);%Basketball drill3 small
 %Ximage=imcrop(Z,[4 100 127 127]);%race horse class C1
% Ximage=imcrop(Z,[200 108 127 127]);%race horse class C2
%Ximage=imcrop(Z,[4 100 63 63]);%race horse class C1 SMALL
%Ximage=imcrop(Z,[200 108 63 63]);%race horse class C2 small
 %Ximage=imcrop(Z,[915 440 127 127]);% parkscene1
% Ximage=imcrop(Z,[915 440 63 63]);% parkscene1 small
% Ximage=imcrop(Z,[1260 716 127 127]); % parkscene2
%Ximage=imcrop(Z,[1260 716 63 63]); % parkscene2 small
% Ximage=imcrop(Z,[750 470 127 127]);%BQTerrace1
%Ximage=imcrop(Z,[750 470 63 63]);%BQTerrace1 small
% Ximage=imcrop(Z,[273 800 127 127]);%BQTerrace2
%Ximage=imcrop(Z,[273 800 63 63]);%BQTerrace2 small
% Ximage=imcrop(Z,[570 1100 127 127]);%SteamLocomotiveTrain1
% Ximage=imcrop(Z,[614 1221 127 127]);%SteamLocomotiveTrain2
%Ximage=imcrop(Z,[570 1100 63 63]);%SteamLocomotiveTrain
%Ximage=imcrop(Z,[444 264 127 127]);%Slide editing
%Ximage=imcrop(Z,[444 264 63 63]);%Slide editing

%X=X';


%im=Z;
im=Ximage;

%READ SEQUENCE
%im=imread('NCI01_J2K_region.tif');

%USE ONLY ONE COLOR COMPONENT - in this case, we use the R component
frame=im(:,:,1);
frame=double(frame);
rf=size(frame,1);
cf=size(frame,2);
%clear im;



% USE INTRA PREDICTION ON THE COLOR COMPONENT
%PROCESS COLOR COMPONENT BLOCK BY BLOCK. BLOCKS ARE OF SIZE 8X8
rb=8;
cb=8;
DCVal=128;
a=0;
aa=0;
count=1;
for rBlock=1:1:(rf/rb)
  %  for rBlock=1:1:1
   for cBlock=1:1:(cf/cb)
  % for cBlock=1:1:8
        fprintf('\nrBlock is %d and cBlock is %d',rBlock,cBlock);
        block=frame(((rBlock-1)*rb)+1:rBlock*rb,((cBlock-1)*cb)+1:cBlock*cb);
        
        %REFERENCE SAMPLES ABOVE
        if rBlock==1
            refAb=zeros(1,cb*3)+DCVal;
            blockAb1=ones(8,8)*DCVal;
            blockAb2=ones(8,8)*DCVal;
        else
            blockAb1=frame(((rBlock-2)*rb)+1:(rBlock-1)*rb,((cBlock-1)*cb)+1:cBlock*cb); %IMMEDIATELY ABOVE
            if cBlock>1 %ABOVE REFERENCE ARRAY CONTAINS THE CORNER REFERENCE
                blockAb2=frame(((rBlock-2)*rb)+1:(rBlock-1)*rb,((cBlock-2)*cb)+1:(cBlock-1)*cb); %ABOVE AND TO THE LEFT
            else
                blockAb2(rb,1:cb)=blockAb1(rb,1);
            end
            if cBlock<(cf/cb)
                blockAb3=frame(((rBlock-2)*rb)+1:(rBlock-1)*rb,((cBlock)*cb)+1:(cBlock+1)*cb); %ABOVE AND TO THE RIGHT
            else
                blockAb3(rb,1:cb)=blockAb1(rb,cb);
            end
            refAb=[blockAb2(rb,:), blockAb1(rb,:), blockAb3(rb,:)];
        end
        
        %REFERENCE SAMPLES TO THE LEFT
        if cBlock==1
            refLe=zeros((rb*2),1)+DCVal;
            blockLe=ones(8,8)*DCVal;
        else
            blockLe=frame(((rBlock-1)*rb)+1:(rBlock)*rb,((cBlock-2)*cb)+1:(cBlock-1)*cb); %IMMEDIATELY LEFT
            padLe=zeros(cb,1)+blockLe(rb,cb);
            refLe= [blockLe(:,cb); padLe];
        end
        epsilon1=1E-10;
        Block_Ref_3D=cat(3,blockAb1,blockAb2,blockLe);
        Block_Ref_3D_temp = abs(Block_Ref_3D);
        Amax = max(Block_Ref_3D_temp(:));
        Amin = min(Block_Ref_3D_temp(:));
        Range = Amax - Amin;
        Anrm = ((Block_Ref_3D_temp - Amin)/(Range + epsilon1));
        eval(['Block_Ref_3D' num2str(count) ' =Anrm']);
       % eval(['Block_Ref_3D' num2str(count) ' =Block_Ref_3D']);
        
        %COMPUTE ACTUAL RESIDUAL USING THE BEST PREDICTION MODE
      %  [resBlock, preBlock, mode]=dirPred(block, refAb, refLe);
%         [resBlock, preBlock, mode]=MY_dirPred(block, refAb, refLe);
      %   [resBlock, preBlock, mode]=dirPred_DC_Planar(block, refAb, refLe);
           [resBlock, preBlock, mode]=dirPred_DC_Planar_all(block, refAb, refLe);
%resBlock=resBlock-mean(mean(resBlock));
            resBlock_temp_1=resBlock;
            eval(['resBlock_temp' num2str(a+1) ' =resBlock_temp_1']);
            a=a+1;
      %eResBlock = MY_estBlocks7(refAb, refLe, rb, cb, mode);
   
         
        
         %REFERENCE SAMPLES ABOVE for Actual Residual for TM 
  
         if rBlock==1
            refAbab=zeros(4,cb*3);
            RefAboveTM= refAbab(:,9:2*cb);
            blockAb1ab=ones(8,8)*DCVal;
            blockAb2ab=ones(8,8)*DCVal;
        else
            blockAb1ab=rFrame(((rBlock-2)*rb)+1:(rBlock-1)*rb,((cBlock-1)*cb)+1:cBlock*cb); %IMMEDIATELY ABOVE
           
            if cBlock>1 %ABOVE REFERENCE ARRAY CONTAINS THE CORNER REFERENCE
                blockAb2ab=rFrame(((rBlock-2)*rb)+1:(rBlock-1)*rb,((cBlock-2)*cb)+1:(cBlock-1)*cb); %ABOVE AND TO THE LEFT
            else
                blockAb2ab(rb,1:cb)=blockAb1ab(rb,1);
            end
            if cBlock<(cf/cb)
                blockAb3ab=rFrame(((rBlock-2)*rb)+1:(rBlock-1)*rb,((cBlock)*cb)+1:(cBlock+1)*cb); %ABOVE AND TO THE RIGHT
            else
                blockAb3ab(rb,1:cb)=blockAb1ab(rb,cb);
            end
          %  refAbab=[blockAb2ab(rb,:), blockAb1ab(rb,:), blockAb3ab(rb,:)];
          refAbab=[blockAb2ab(5:rb,:), blockAb1ab(5:rb,:), blockAb3ab(5:rb,:)];
           
            RefAboveTM= blockAb1ab(5:rb,:);  
         end
         
        % 
          %REFERENCE SAMPLES TO THE LEFT for Actual residual for TM
          
          if cBlock==1
          %  refLeLe=zeros((rb*2),2);
            refLeLe=zeros((rb*2),4);
            blockLeLe=ones(8,8)*DCVal;
        else
            blockLeLe=rFrame(((rBlock-1)*rb)+1:(rBlock)*rb,((cBlock-2)*cb)+1:(cBlock-1)*cb); %IMMEDIATELY LEFT
          % padLe=zeros(cb,1)+blockLeLe(rb,cb);
          %  refLeLe= [blockLeLe(:,cb); padLe];
          %   padLe=zeros(cb,2)+blockLeLe(rb,7:cb);
              padLe=zeros(cb,4)+blockLeLe(rb,5:cb);
             refLeLe= [blockLeLe(:,5:cb); padLe];
          end
        
        ResBlock_Ref_3D=cat(3,blockAb1,blockAb2,blockLe);
        eval(['ResBlock_Ref_3D' num2str(count) ' =ResBlock_Ref_3D']);
        
        % NEW CODE 1st May 2023 START
        
        ResBlock_Ref_3D_Res=cat(3,blockAb1ab,blockAb2ab,blockLeLe);
        eval(['ResBlock_Ref_3D_Res' num2str(count) ' =ResBlock_Ref_3D_Res']);
        
        % Above Cov
        mn_blockAb1ab=mean(mean(blockAb1ab));
        blockAb1ab_2=blockAb1ab-mn_blockAb1ab;
        blockAb1ab_3=blockAb1ab_2(:);
        R_blockAb1ab = (blockAb1ab_3*blockAb1ab_3')/((cb*rb)-1);

        % Corner Cov
        mn_blockAb2ab=mean(mean(blockAb2ab));
        blockAb2ab_2=blockAb2ab-mn_blockAb2ab;
        blockAb2ab_3=blockAb2ab_2(:);
        R_blockAb2ab = (blockAb2ab_3*blockAb2ab_3')/((cb*rb)-1);

        % Left Cov
        mn_blockLeLe=mean(mean(blockLeLe));
        blockLeLe_2=blockLeLe-mn_blockLeLe;
        blockLeLe_3=blockLeLe_2(:);
        R_blockLeLe = (blockLeLe_3*blockLeLe_3')/((cb*rb)-1); 
        
        
        
        eval(['R_blockAb1ab' num2str(count) ' =R_blockAb1ab']);
        eval(['R_blockAb2ab' num2str(count) ' =R_blockAb2ab']);
        eval(['R_blockLeLe' num2str(count) ' =R_blockLeLe']);
        
        % Average of Above, Corner and Left Cov
        epsilon_ave=1E-10;
        R_block_ACL_Cov_Average = (R_blockAb1ab+R_blockAb2ab+R_blockLeLe)/3;
        R_block_ACL_Cov_Average( R_block_ACL_Cov_Average == 0 ) = epsilon_ave;
        eval(['R_block_ACL_Cov_Average' num2str(count) ' =R_block_ACL_Cov_Average']);
        
        [Ucov,Scov,Vcov] = svd(R_block_ACL_Cov_Average);
        R_block_ACL_Cov_Average_SVD = Vcov*inv(Scov)*Ucov';
        eval(['R_block_ACL_Cov_Average_SVD' num2str(count) ' =R_block_ACL_Cov_Average_SVD']);
        
        
        R_block_ACL_Cov_Average_SVD_R = reshape(R_block_ACL_Cov_Average_SVD,1,64*64);
        Precision_Final(count,:)=R_block_ACL_Cov_Average_SVD_R;
        
        
        % NEW CODE 1st May 2023 END
        
        
        % NEW CODE 18th May 2023 START for GBT-ONL ground truth
        
        Avg_Ref_New = (blockAb1ab + blockAb2ab +blockLeLe)/3;
        
        Avg_Ref_New_Rshape = reshape(Avg_Ref_New,1,8*8);
        Avg_Ref_New_Final(count,:)=Avg_Ref_New_Rshape;
        
        [Lrefavg] = eigenLapAll(Avg_Ref_New);
        eval(['Lrefavg' num2str(count) ' =Lrefavg']);
        Lrefavg_Rshape = reshape(Lrefavg,1,64*64);
        Lrefavg_Final(count,:)=Lrefavg_Rshape;
        
        % NEW CODE 18th May 2023 END
        
        
   
        
        
      %   refabab=refAbab(cb+1:cb*2);
       % ex=[refabab;resBlock];
       %  refLeLe=refLeLe(1:rb+1);
       %  patch= [refLeLe ex];
      refabab= refAbab(:,cb+1:cb*2);
      expatch=[refabab;resBlock];
      refableft=refAbab(:,cb-3:cb);
      refleadd=[refableft;refLeLe];
       reflesub=refleadd(1:12,:);
       RefLeftTM=reflesub;
       patchfinal=[reflesub,expatch];
     %  onlyTM2=patchfinal;
      % AA=onlyTM2(5:12,5:12);
     %  onlyTM2(5:12,5:12)=0;
       
       
     % refabextra=(refAbab(:,((cb*2)+1):((cb*2)+2)));
     %  padRight=zeros(8,2);
     %  refextraright=[refabextra;padRight];
     %  patchfinal=[patchnotfinal,refextraright];
    % sx= size(RefAboveTM);
      % sy=size(RefLeftTM);
     %  maxam= max(sx(1),sy(1));

     % onlyTM= [[RefLeftTM;zeros(abs([maxam,0]-sy))],[RefAboveTM;zeros(abs([maxam,0]-sx))]] ;
       
    %   [V_graph2,D_graph2]= eigenLap2onlyTM(onlyTM2);
       
   %    D_Graph_sum= sum (sum (D_graph));
     %  D_Graph_sum1=zeros(1,cBlock*rBlock);
    %  for f=1:1:cBlock*rBlock
    %   D_Graph_sum1(1,cBlock*rBlock) = D_Graph_sum;
     % end 
         %CREATE  Patch FRAME
         rb_p=rb+4;
         cb_p=cb+4;
         for k=1:1:rb_p
             for j=1:1:cb_p
         
          patchFrame(((rBlock-1)*rb_p)+1:rBlock*rb_p,((cBlock-1)*cb_p)+1:cBlock*cb_p) = patchfinal;
        %  onlyTMFrame(((rBlock-1)*rb_p)+1:rBlock*rb_p,((cBlock-1)*cb_p)+1:cBlock*cb_p) = onlyTM2;
             end
         end
         
        n=size(patchFrame,1);
 m=rb+4;
 row_patchFrame=[];
% onlyrow_patchFrame=[];
 norows=n/m;
 
 for s= 1:norows
row_patchFrame= [row_patchFrame,patchFrame(((s-1)*m+1):((s-1)*m+m),:)];
%onlyrow_patchFrame= [onlyrow_patchFrame,onlyTMFrame(((s-1)*m+1):((s-1)*m+m),:)];
 end
 
 
      rTMA=1;
      rTMAA=1;
%{
for f=1:12:12*cBlock*rBlock
   % onlyTM2=onlyrow_patchFrame(:,f:(f+11));
  %  [V_graph,D_graph]= eigenLap2onlyTM(onlyTM2);
  %  D_Graph_sum= sum (sum (D_graph));
% D_Graph_sum1(1,rTMAA) = D_Graph_sum;
D_Graph_M(1,rTMAA).M = D_graph;

%TMBlocksA(1,rTMAA).B = AA;
%for m=5:12:(cb+4)*cBlock*rBlock

    refabrowpatch1A=row_patchFrame(1,f+4:(f+11));
    refabrowpatch2A=row_patchFrame(2,f+4:(f+11));
    refabrowpatch3A=row_patchFrame(3,f+4:(f+11));
    refabrowpatch4A=row_patchFrame(4,f+4:(f+11));
    refLecolpatch1A=row_patchFrame(1:12,f+3);
    refLecolpatch2A=row_patchFrame(1:12,f+2);
    refLecolpatch3A=row_patchFrame(1:12,f+1);
    refLecolpatch4A=row_patchFrame(1:12,f);
    
    refAbTM_A(1,rTMAA).R1 = refabrowpatch1A;
    refAbTM_A(1,rTMAA).R2 = refabrowpatch2A;
    refAbTM_A(1,rTMAA).R3 = refabrowpatch3A;
    refAbTM_A(1,rTMAA).R4 = refabrowpatch4A;
    refLeTM_A(1,rTMAA).C1 = refLecolpatch1A;
    refLeTM_A(1,rTMAA).C2 = refLecolpatch2A;
    refLeTM_A(1,rTMAA).C3 = refLecolpatch3A;
    refLeTM_A(1,rTMAA).C4 = refLecolpatch4A;

    row_patchFrameBlockA=row_patchFrame(5:12,f+4:f+11);
    TMBlocksA(1,rTMAA).B = row_patchFrameBlockA;

if rTMAA<=5 && rBlock==1
           eResBlock_graph = MY_estBlocks7(refAb, refLe, rb, cb, mode); % first 5 blocks of Predicted residual by PI
           TMBlocksNew(1,rTMA).B = eResBlock_graph;
    else
        if rTMAA>=6
        for rA=1:1:rTMAA-1
         %   rAdiffsq(1,rA)=(imabsdiff(D_Graph_sum1(1,rTMAA),D_Graph_sum1(1,rA))).^2;
            rAdiffsq_M(1,rA)=sum(sum((D_Graph_M(1,rTMAA).M-D_Graph_M(1,rA).M).^2)); 
        end        
    %    yA=rAdiffsq;
        yA_M=rAdiffsq_M;
        nA=5;
   %     [ysA, index] = sort(yA);
   %     qA=index(1:nA);
        [ysA_M, index] = sort(yA_M);
        qA_M=index(1:nA);
        
        
        
            for zA=1:1:nA
            corrbloackA=TMBlocksA(1,qA_M(zA)).B;
            eval(['PA' num2str(zA) ' =corrbloackA']);
            CMAT_A=[refLeTM_A(1,qA_M(zA)).C4;refLeTM_A(1,qA_M(zA)).C3;refLeTM_A(1,qA_M(zA)).C2;refLeTM_A(1,qA_M(zA)).C1;refAbTM_A(1,qA_M(zA)).R1';refAbTM_A(1,qA_M(zA)).R2';refAbTM_A(1,qA_M(zA)).R3';refAbTM_A(1,qA_M(zA)).R4'];
            eval(['CORRMAT_A' num2str(zA) ' =CMAT_A']);
          %  newsum=newsum1+newsum;
            
            end
        
                    CORRMAT_A=[CORRMAT_A1 CORRMAT_A2 CORRMAT_A3 CORRMAT_A4 CORRMAT_A5];
           % newsum=newsum/5;
           % TMBlocksNew(1,r).B = newsum;
            TARTEMP_A=[refLeTM_A(1,rTMAA).C4;refLeTM_A(1,rTMAA).C3;refLeTM_A(1,rTMAA).C2;refLeTM_A(1,rTMAA).C1;refAbTM_A(1,rTMAA).R1';refAbTM_A(1,rTMAA).R2';refAbTM_A(1,rTMAA).R3';refAbTM_A(1,rTMAA).R4'];
            beqA=1;
            beqbA=2;
            AeqA=ones(1,5);
            lbA=zeros(5,1);
            ubA=ones(5,1);
            AA=AeqA;
            
            X_graph = lsqlin(CORRMAT_A,TARTEMP_A,AA,beqbA,AeqA,beqA,lbA, ubA); % optimization funtion to calculate optimized weight
            %  X = lsqlin(abs(CORRMAT),abs(TARTEMP),A,beqb,Aeq,beq,0,Inf);
            PBlock_A= PA1*X_graph(1)+PA2*X_graph(2)+PA3*X_graph(3)+PA4*X_graph(4)+PA5*X_graph(5);
           % eResBlock = newsum;
            eResBlock_graph = PBlock_A;  % estimated/predicted residual block using TM
XMAT1A=zeros(5,(rf/rb)*(cf/cb)-5);
XMATA(:,rTMAA-5) = X_graph;
XMATA=XMAT1A+XMATA;
        
         end
        
end

rTMAA=rTMAA+1;
 %  end
end
%}

for m=5:12:(cb+4)*cBlock*rBlock
    refabrowpatch1=row_patchFrame(1,m:(m+7));
    refabrowpatch2=row_patchFrame(2,m:(m+7));
    refabrowpatch3=row_patchFrame(3,m:(m+7));
    refabrowpatch4=row_patchFrame(4,m:(m+7));
    refLecolpatch1=row_patchFrame(1:12,m-1);
    refLecolpatch2=row_patchFrame(1:12,m-2);
    refLecolpatch3=row_patchFrame(1:12,m-3);
    refLecolpatch4=row_patchFrame(1:12,m-4);
    row_patchFrameBlock=row_patchFrame(5:12,m:m+7);
    refAbTM(1,rTMA).R1 = refabrowpatch1;
    refAbTM(1,rTMA).R2 = refabrowpatch2;
    refAbTM(1,rTMA).R3 = refabrowpatch3;
    refAbTM(1,rTMA).R4 = refabrowpatch4;
    refLeTM(1,rTMA).C1 = refLecolpatch1;
    refLeTM(1,rTMA).C2 = refLecolpatch2;
    refLeTM(1,rTMA).C3 = refLecolpatch3;
    refLeTM(1,rTMA).C4 = refLecolpatch4;
    TMBlocks(1,rTMA).B = row_patchFrameBlock;
    %D_Graph_sum1(1,rTMA) = D_Graph_sum;
    
    if rTMA<=5 && rBlock==1
           eResBlock = MY_estBlocks7(refAb, refLe, rb, cb, mode); % first 5 blocks of Predicted residual by PI
           TMBlocksNew(1,rTMA).B = eResBlock;
           
        mneR=mean(mean(eResBlock));
        resBlock2eR=eResBlock-mneR;
        %resBlock3=reshape(resBlock2, cb*rb,1);
        resBlock3eR=resBlock2eR(:);
        ReR = (resBlock3eR*resBlock3eR')/((cb*rb)-1);
        ReRshape = reshape(ReR,1,64*64);

%%%% Adj matrix start
rba=size(ReR,1);
cba=size(ReR,2);

ReRAdj=zeros(rba,cba);

for ia=1:1:rba-1
    for ja=1:1:cba-1
        val=sum((ReR(:,ia)-ReR(:,ja+1)).^2);
        if ia==ja
          ReRAdj(ia,ja)=0;
        end
        ReRAdj(ia,ja+1)=val;
        ReRAdj(ja+1,ia)=val;
     end
end
ReRAdjshape = reshape(ReRAdj,1,64*64);
%%%% Adj matrix end

    else
        if rTMA>=6
        for r1=1:1:rTMA-1
        Abdiff1=sum(imabsdiff(refAbTM(1,rTMA).R1,refAbTM(1,r1).R1));
        Abdiff2=sum(imabsdiff(refAbTM(1,rTMA).R2,refAbTM(1,r1).R2));
        Abdiff3=sum(imabsdiff(refAbTM(1,rTMA).R3,refAbTM(1,r1).R3));
        Abdiff4=sum(imabsdiff(refAbTM(1,rTMA).R4,refAbTM(1,r1).R4));
        Abdiff=Abdiff1+Abdiff2+Abdiff3+Abdiff4;
        Lediff1=sum(imabsdiff(refLeTM(1,rTMA).C1,refLeTM(1,r1).C1));
        Lediff2=sum(imabsdiff(refLeTM(1,rTMA).C2,refLeTM(1,r1).C2));
        Lediff3=sum(imabsdiff(refLeTM(1,rTMA).C3,refLeTM(1,r1).C3));
        Lediff4=sum(imabsdiff(refLeTM(1,rTMA).C4,refLeTM(1,r1).C4));
        Lediff=Lediff1+Lediff2+Lediff3+Lediff4;
        minAbdiff(1,r1)=Abdiff;
        minLediff(1,r1)=Lediff;
        sumdiff(1,r1)=Abdiff+Lediff;
        
        
        
        end        
        y1=sumdiff;
        n=5;
        [ys, index] = sort(y1);
        q=index(1:n);
       % re = x1(sort(index(1:n)));
       % [p,q]=find(sumdiff==xs)
        %{
        val=zeros(n,1);
        for i=1:1:n
            newsum=zeros(rb,cb);
            [val(i),index]=min(x1);
            [p,q]=find(sumdiff==val);
            x1(index)=[];
        end
        %}
           % newsum=zeros(rb,cb);
            for z1=1:1:n
            corrbloack=TMBlocks(1,q(z1)).B;
            eval(['P' num2str(z1) ' =corrbloack']);
            CMAT=[refLeTM(1,q(z1)).C4;refLeTM(1,q(z1)).C3;refLeTM(1,q(z1)).C2;refLeTM(1,q(z1)).C1;refAbTM(1,q(z1)).R1';refAbTM(1,q(z1)).R2';refAbTM(1,q(z1)).R3';refAbTM(1,q(z1)).R4'];
            eval(['CORRMAT' num2str(z1) ' =CMAT']);
          %  newsum=newsum1+newsum;
            end
            
        P1mn=mean(mean(P1));
        P1resBlock2=P1-P1mn;
        %resBlock3=reshape(resBlock2, cb*rb,1);
        P1resBlock3=P1resBlock2(:);
        P1Res = (P1resBlock3*P1resBlock3')/((cb*rb)-1);
     
                P2mn=mean(mean(P2));
        P2resBlock2=P2-P2mn;
        %resBlock3=reshape(resBlock2, cb*rb,1);
        P2resBlock3=P2resBlock2(:);
        P2Res = (P2resBlock3*P2resBlock3')/((cb*rb)-1);
        
                P3mn=mean(mean(P3));
        P3resBlock2=P3-P3mn;
        %resBlock3=reshape(resBlock2, cb*rb,1);
        P3resBlock3=P3resBlock2(:);
        P3Res = (P3resBlock3*P3resBlock3')/((cb*rb)-1);
        
                P4mn=mean(mean(P4));
        P4resBlock2=P4-P4mn;
        %resBlock3=reshape(resBlock2, cb*rb,1);
        P4resBlock3=P4resBlock2(:);
        P4Res = (P4resBlock3*P4resBlock3')/((cb*rb)-1);
        
                P5mn=mean(mean(P5));
        P5resBlock2=P5-P5mn;
        %resBlock3=reshape(resBlock2, cb*rb,1);
        P5resBlock3=P5resBlock2(:);
        P5Res = (P5resBlock3*P5resBlock3')/((cb*rb)-1);
            
            CORRMAT=[CORRMAT1 CORRMAT2 CORRMAT3 CORRMAT4 CORRMAT5];
           % newsum=newsum/5;
           % TMBlocksNew(1,r).B = newsum;
            TARTEMP=[refLeTM(1,rTMA).C4;refLeTM(1,rTMA).C3;refLeTM(1,rTMA).C2;refLeTM(1,rTMA).C1;refAbTM(1,rTMA).R1';refAbTM(1,rTMA).R2';refAbTM(1,rTMA).R3';refAbTM(1,rTMA).R4'];
            beq=1;
            beqb=2;
            Aeq=ones(1,5);
            lb=zeros(5,1);
            ub=ones(5,1);
            A=Aeq;
            
            X = lsqlin(CORRMAT,TARTEMP,A,beqb,Aeq,beq,lb, ub); % optimization funtion to calculate optimized weight
            %  X = lsqlin(abs(CORRMAT),abs(TARTEMP),A,beqb,Aeq,beq,0,Inf);
            PBlock= P1*X(1)+P2*X(2)+P3*X(3)+P4*X(4)+P5*X(5);
            S=P1Res*X(1)+P2Res*X(2)+P3Res*X(3)+P4Res*X(4)+P5Res*X(5);
            S_F=S(:);
            [Sadj] = eigenLap2_Sadj(S);
           % eResBlock = newsum;
            eResBlock = PBlock;  % estimated/predicted residual block using TM
            
        mneR=mean(mean(eResBlock));
        resBlock2eR=eResBlock-mneR;
        %resBlock3=reshape(resBlock2, cb*rb,1);
        resBlock3eR=resBlock2eR(:);
        ReR = (resBlock3eR*resBlock3eR')/((cb*rb)-1);
        ReRshape = reshape(ReR,1,64*64);
        
%%%% Adj matrix start
rba=size(ReR,1);
cba=size(ReR,2);

ReRAdj=zeros(rba,cba);

for ia=1:1:rba-1
    for ja=1:1:cba-1
        val=sum((ReR(:,ia)-ReR(:,ja+1)).^2);
        if ia==ja
          ReRAdj(ia,ja)=0;
        end
        ReRAdj(ia,ja+1)=val;
        ReRAdj(ja+1,ia)=val;
     end
end
ReRAdjshape = reshape(ReRAdj,1,64*64);
%%%% Adj matrix end
        
XMAT1=zeros(5,(rf/rb)*(cf/cb)-5);
XMAT(:,rTMA-5) = X;
XMAT=XMAT1+XMAT;
        end
    end
    dis=eResBlock;
    dis1=ReR;
    disAdj=ReRAdj;
    eval(['ReR' num2str(rTMA) ' =ReR']);
    eval(['ReRAdj' num2str(rTMA) ' =ReRAdj']);
    eval(['ReRAdjshape' num2str(rTMA) ' =ReRAdjshape']);
    eval(['ReRshape' num2str(rTMA) ' =ReRshape']);
    eres_Cov_Final(rTMA,:)=ReRshape;
    res_Adj_Final(rTMA,:)=ReRAdjshape;
    rTMA=rTMA+1;
end

%toc
%tt1=toc;

%%%%%%%%%------------ TEMPLATE MATCHING (residual domain) END--------------%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%------------ TEMPLATE POOLING (residual domain) START--------------%%%%%%%%%%%%%%%%%%%%


         
  
         if rBlock==1
            refAbab=zeros(4,cb*3);
        else
            blockAb1ab=rFrame(((rBlock-2)*rb)+1:(rBlock-1)*rb,((cBlock-1)*cb)+1:cBlock*cb); %IMMEDIATELY ABOVE
            if cBlock>1 %ABOVE REFERENCE ARRAY CONTAINS THE CORNER REFERENCE
                blockAb2ab=rFrame(((rBlock-2)*rb)+1:(rBlock-1)*rb,((cBlock-2)*cb)+1:(cBlock-1)*cb); %ABOVE AND TO THE LEFT
            else
                blockAb2ab(rb,1:cb)=blockAb1ab(rb,1);
            end
            if cBlock<(cf/cb)
                blockAb3ab=rFrame(((rBlock-2)*rb)+1:(rBlock-1)*rb,((cBlock)*cb)+1:(cBlock+1)*cb); %ABOVE AND TO THE RIGHT
            else
                blockAb3ab(rb,1:cb)=blockAb1ab(rb,cb);
            end
          %  refAbab=[blockAb2ab(rb,:), blockAb1ab(rb,:), blockAb3ab(rb,:)];
          refAbab=[blockAb2ab(5:rb,:), blockAb1ab(5:rb,:), blockAb3ab(5:rb,:)];
         
         end
          %REFERENCE SAMPLES TO THE LEFT for Actual residual for TM
          
        if cBlock==1
          %  refLeLe=zeros((rb*2),2);
            refLeLe=zeros((rb*2),4);
        else
            blockLeLe=rFrame(((rBlock-1)*rb)+1:(rBlock)*rb,((cBlock-2)*cb)+1:(cBlock-1)*cb); %IMMEDIATELY LEFT
          % padLe=zeros(cb,1)+blockLeLe(rb,cb);
          %  refLeLe= [blockLeLe(:,cb); padLe];
          %   padLe=zeros(cb,2)+blockLeLe(rb,7:cb);
              padLe=zeros(cb,4)+blockLeLe(rb,5:cb);
             refLeLe= [blockLeLe(:,5:cb); padLe];
        end
        

      %   refabab=refAbab(cb+1:cb*2);
       % ex=[refabab;resBlock];
       %  refLeLe=refLeLe(1:rb+1);
       %  patch= [refLeLe ex];
      refabab= refAbab(:,cb+1:cb*2);
      expatch=[refabab;resBlock];
      refableft=refAbab(:,cb-3:cb);
      refleadd=[refableft;refLeLe];
       reflesub=refleadd(1:12,:);
       
       patchfinal=[reflesub,expatch];
     % refabextra=(refAbab(:,((cb*2)+1):((cb*2)+2)));
     %  padRight=zeros(8,2);
     %  refextraright=[refabextra;padRight];
     %  patchfinal=[patchnotfinal,refextraright];


         %CREATE  Patch FRAME
         rb_p=rb+4;
         cb_p=cb+4;
         for k=1:1:rb_p
             for j=1:1:cb_p
         
          patchFrame(((rBlock-1)*rb_p)+1:rBlock*rb_p,((cBlock-1)*cb_p)+1:cBlock*cb_p) = patchfinal;
             end
         end
         
        n=size(patchFrame,1);
 m=rb+4;
 row_patchFrame=[];
 norows=n/m;
 
 for s= 1:norows
row_patchFrame= [row_patchFrame,patchFrame(((s-1)*m+1):((s-1)*m+m),:)];
 end
 
 
      rn=1;

for m=5:12:(cb+4)*cBlock*rBlock
    refabrowpatch1=row_patchFrame(1,m:(m+7));
    refabrowpatch2=row_patchFrame(2,m:(m+7));
    refabrowpatch3=row_patchFrame(3,m:(m+7));
    refabrowpatch4=row_patchFrame(4,m:(m+7));
    refLecolpatch1=row_patchFrame(1:12,m-1);
    refLecolpatch2=row_patchFrame(1:12,m-2);
    refLecolpatch3=row_patchFrame(1:12,m-3);
    refLecolpatch4=row_patchFrame(1:12,m-4);
    row_patchFrameBlock=row_patchFrame(5:12,m:m+7);
    refAbTM(1,rn).R1 = refabrowpatch1;
    refAbTM(1,rn).R2 = refabrowpatch2;
    refAbTM(1,rn).R3 = refabrowpatch3;
    refAbTM(1,rn).R4 = refabrowpatch4;
    refLeTM(1,rn).C1 = refLecolpatch1;
    refLeTM(1,rn).C2 = refLecolpatch2;
    refLeTM(1,rn).C3 = refLecolpatch3;
    refLeTM(1,rn).C4 = refLecolpatch4;
    TMBlocksNLM(1,rn).B = row_patchFrameBlock;
    TM_Col(1,rn).TC = [refLeTM(1,rn).C4;refLeTM(1,rn).C3;refLeTM(1,rn).C2;refLeTM(1,rn).C1;refAbTM(1,rn).R1';refAbTM(1,rn).R2';refAbTM(1,rn).R3';refAbTM(1,rn).R4'];
    
    if rn<=5 && rBlock==1
         %  eResBlock = MY_estBlocks7(refAb, refLe, rb, cb, mode); % first 5 blocks of Predicted residual by PI
           PNLMBlockFinal = MY_estBlocks7(refAb, refLe, rb, cb, mode); % estimated/predicted residual block using PI;
           %PNLMBlockFinal = eResBlock;
           TMBlocksNew1(1,rn).B = eResBlock;
    else
        if rn>=6
        
        for r11=1:1:rn-1
        NLM(1,r11).SS =(imabsdiff(TM_Col(1,rn).TC,TM_Col(1,r11).TC)).^2;
        NLM(1,r11).SSADD = sum(NLM(1,r11).SS);
        %NLM(1,r11).SSADD = sqrt(NLM(1,r11).SSADD);
        add(r11)=NLM(1,r11).SSADD;
        end  
        
        %maxadd=max(add);
        maxadd=std(add);
        SSADDMAXDIV = add/maxadd;
        SSADDMAXDIVEXP = exp(-SSADDMAXDIV);
        SSADDMAXDIVEXPW = SSADDMAXDIVEXP/sum(SSADDMAXDIVEXP);
        PNLMBlockF=zeros(rb,cb);
        
        u=rn;
        for nl=1:1:u-1
        Q=TMBlocksNLM(1,nl).B;
        PNLMBlock = Q*SSADDMAXDIVEXPW(nl);
        PNLMBlockF = PNLMBlockF + PNLMBlock;
        PNLMBlockFinal = PNLMBlockF;
        end
        

        end
    end
    rn=rn+1;
end

%toc
%tt=toc;
%%%%%%%%%------------ TEMPLATE POOLING (residual domain) END--------------%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%------------ TEMPLATE MATCHING (pixel domain + graph domain) START--------------%%%%%%%%%%%%%%%%%%%%
 [V_act,D_act] = eigenLap2(block);
 block_all=block(:);
  coeff_block= (V_act')*block_all;
  coeff_block_reshape=reshape(coeff_block,rb,cb);
  maxCoeff_block=max(max(abs(coeff_block)));
  minCoeff_block=min(min(abs(coeff_block))); 
  step=0.25;
  %step=1;
    for th=maxCoeff_block:-step:minCoeff_block-step
            
            mask=abs(coeff_block_reshape)>th;
            tmpBlock=mask.*coeff_block_reshape;
            eTmpBlock_block=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
             tmpblock_actual = reshape(tmpBlock,rb*cb,1);
                
            recons_block=(V_act)* tmpblock_actual;
            
           recons_block_final = reshape(recons_block,rb,cb); %Reconstructed block
    end
      %REFERENCE SAMPLES ABOVE for Actual Residual for TM TYPE B
     if rBlock==1
            refAbabTM=zeros(4,cb*3);
        else
            blockAb1abTM=reConsFrame_TM_B(((rBlock-2)*rb)+1:(rBlock-1)*rb,((cBlock-1)*cb)+1:cBlock*cb); %IMMEDIATELY ABOVE
            if cBlock>1 %ABOVE REFERENCE ARRAY CONTAINS THE CORNER REFERENCE
                blockAb2abTM=reConsFrame_TM_B(((rBlock-2)*rb)+1:(rBlock-1)*rb,((cBlock-2)*cb)+1:(cBlock-1)*cb); %ABOVE AND TO THE LEFT
            else
                blockAb2abTM(rb,1:cb)=blockAb1abTM(rb,1);
            end
            if cBlock<(cf/cb)
                blockAb3abTM=reConsFrame_TM_B(((rBlock-2)*rb)+1:(rBlock-1)*rb,((cBlock)*cb)+1:(cBlock+1)*cb); %ABOVE AND TO THE RIGHT
            else
                blockAb3abTM(rb,1:cb)=blockAb1abTM(rb,cb);
            end
          %  refAbab=[blockAb2ab(rb,:), blockAb1ab(rb,:), blockAb3ab(rb,:)];
          refAbabTM=[blockAb2abTM(5:rb,:), blockAb1abTM(5:rb,:), blockAb3abTM(5:rb,:)];
         end
          %REFERENCE SAMPLES TO THE LEFT for Actual residual for TM TYPE B
          
          if cBlock==1
          %  refLeLe=zeros((rb*2),2);
            refLeLeTM=zeros((rb*2),4);
        else
            blockLeLeTM=reConsFrame_TM_B(((rBlock-1)*rb)+1:(rBlock)*rb,((cBlock-2)*cb)+1:(cBlock-1)*cb); %IMMEDIATELY LEFT
          % padLe=zeros(cb,1)+blockLeLe(rb,cb);
          %  refLeLe= [blockLeLe(:,cb); padLe];
          %   padLe=zeros(cb,2)+blockLeLe(rb,7:cb);
              padLeTM=zeros(cb,4)+blockLeLeTM(rb,5:cb);
             refLeLeTM= [blockLeLeTM(:,5:cb); padLeTM];
        end
        

      %   refabab=refAbab(cb+1:cb*2);
       % ex=[refabab;resBlock];
       %  refLeLe=refLeLe(1:rb+1);
       %  patch= [refLeLe ex];
      refababTM= refAbab(:,cb+1:cb*2);
      expatchTM=[refababTM;recons_block_final];
      refableftTM=refAbabTM(:,cb-3:cb);
      refleaddTM=[refableftTM;refLeLeTM];
       reflesubTM=refleaddTM(1:12,:);
       
       patchfinalTM=[reflesubTM,expatchTM];
     % refabextra=(refAbab(:,((cb*2)+1):((cb*2)+2)));
     %  padRight=zeros(8,2);
     %  refextraright=[refabextra;padRight];
     %  patchfinal=[patchnotfinal,refextraright];
onlyTM2_pixel=patchfinalTM;
      % AA=onlyTM2(5:12,5:12);
       onlyTM2_pixel(5:12,5:12)=0; 
       
       
     % refabextra=(refAbab(:,((cb*2)+1):((cb*2)+2)));
     %  padRight=zeros(8,2);
     %  refextraright=[refabextra;padRight];
     %  patchfinal=[patchnotfinal,refextraright];
    % sx= size(RefAboveTM);
      % sy=size(RefLeftTM);
     %  maxam= max(sx(1),sy(1));

     % onlyTM= [[RefLeftTM;zeros(abs([maxam,0]-sy))],[RefAboveTM;zeros(abs([maxam,0]-sx))]] ;
       
     %  [V_graph2_pixel,D_graph2_pixel]= eigenLap2onlyTM(onlyTM2_pixel);
     
     
  
         %CREATE  Patch FRAME for Reconstructed block
         rb_pTM=rb+4;
         cb_pTM=cb+4;
         for k=1:1:rb_pTM
             for j=1:1:cb_pTM
         
          patchFrameTM(((rBlock-1)*rb_p)+1:rBlock*rb_p,((cBlock-1)*cb_p)+1:cBlock*cb_p) = patchfinalTM;
          onlyTM2_pixel_Frame(((rBlock-1)*rb_p)+1:rBlock*rb_p,((cBlock-1)*cb_p)+1:cBlock*cb_p) = onlyTM2_pixel;
             end
         end
         
        nTM=size(patchFrameTM,1);
 mTM=rb+4;
 row_patchFrameTM=[];
 row_onlyTM2_pixel_Frame=[];
 norowsTM=nTM/mTM;
 
 for sTM= 1:norowsTM
row_patchFrameTM= [row_patchFrameTM,patchFrameTM(((sTM-1)*mTM+1):((sTM-1)*mTM+mTM),:)];
row_onlyTM2_pixel_Frame= [row_onlyTM2_pixel_Frame,onlyTM2_pixel_Frame(((sTM-1)*mTM+1):((sTM-1)*mTM+mTM),:)];
 end

% PENDING                       PENDING                    PENDING
 
       rTMB=1;
       rTMBB=1;
for f=1:12:12*cBlock*rBlock
    onlyTM2_TM=row_onlyTM2_pixel_Frame(:,f:(f+11));
    [V_graph_TM,D_graph_TM]= eigenLap2onlyTM(onlyTM2_TM); %%%%%%%%%%%%%%%%%%--------Template Matching Graph Domain ----------------------- %%%%%%%%%%%%%%%%%%%
 %   D_Graph_sum= sum (sum (D_graph));
% D_Graph_sum1(1,rTMAA) = D_Graph_sum;
D_Graph_TM_M(1,rTMBB).M = D_graph_TM;

%TMBlocksA(1,rTMAA).B = AA;
%for m=5:12:(cb+4)*cBlock*rBlock

    refabrowpatch1B_TM=row_patchFrameTM(1,f+4:(f+11));
    refabrowpatch2B_TM=row_patchFrameTM(2,f+4:(f+11));
    refabrowpatch3B_TM=row_patchFrameTM(3,f+4:(f+11));
    refabrowpatch4B_TM=row_patchFrameTM(4,f+4:(f+11));
    refLecolpatch1B_TM=row_patchFrameTM(1:12,f+3);
    refLecolpatch2B_TM=row_patchFrameTM(1:12,f+2);
    refLecolpatch3B_TM=row_patchFrameTM(1:12,f+1);
    refLecolpatch4B_TM=row_patchFrameTM(1:12,f);
    
    refAbTM_TM_A(1,rTMBB).R1 = refabrowpatch1B_TM;
    refAbTM_TM_A(1,rTMBB).R2 = refabrowpatch2B_TM;
    refAbTM_TM_A(1,rTMBB).R3 = refabrowpatch3B_TM;
    refAbTM_TM_A(1,rTMBB).R4 = refabrowpatch4B_TM;
    refLeTM_TM_A(1,rTMBB).C1 = refLecolpatch1B_TM;
    refLeTM_TM_A(1,rTMBB).C2 = refLecolpatch2B_TM;
    refLeTM_TM_A(1,rTMBB).C3 = refLecolpatch3B_TM;
    refLeTM_TM_A(1,rTMBB).C4 = refLecolpatch4B_TM;

    row_patchFrameBlockB_TM=row_patchFrameTM(5:12,f+4:f+11);
    TMBlocksB_TM(1,rTMBB).B = row_patchFrameBlockB_TM;
    
if rTMBB<=5 && rBlock==1
           eResBlock_graph_TM = MY_estBlocks7(refAb, refLe, rb, cb, mode); % first 5 blocks of Predicted residual by PI
           TMBlocksNew_TM(1,rTMB).B = eResBlock_graph_TM;
    else
        if rTMBB>=6
        for rA_TM=1:1:rTMBB-1
        %    rAdiffsq(1,rA_TM)=(imabsdiff(D_Graph_sum1(1,rTMBB),D_Graph_sum1(1,rA_TM))).^2;
            rAdiffsq_TM_M(1,rA_TM)=sum(sum((D_Graph_TM_M(1,rTMBB).M-D_Graph_TM_M(1,rA_TM).M).^2)); % similarity in graph domain
      
        end        
    %    yA=rAdiffsq;
        yA_TM_M=rAdiffsq_TM_M;
        nA=5;
   %     [ysA, index] = sort(yA);
   %     qA=index(1:nA);
        [ysA_TM_M, index] = sort(yA_TM_M);
        qA_TM_M=index(1:nA);
        
        
        
            for zA_TM=1:1:nA
            corrbloackA_TM=TMBlocksB_TM(1,qA_TM_M(zA_TM)).B;
            eval(['PA_TM' num2str(zA_TM) ' =corrbloackA_TM']);
            CMAT_TM_A=[refLeTM_TM_A(1,qA_TM_M(zA_TM)).C4;refLeTM_TM_A(1,qA_TM_M(zA_TM)).C3;refLeTM_TM_A(1,qA_TM_M(zA_TM)).C2;refLeTM_TM_A(1,qA_TM_M(zA_TM)).C1;refAbTM_TM_A(1,qA_TM_M(zA_TM)).R1';refAbTM_TM_A(1,qA_TM_M(zA_TM)).R2';refAbTM_TM_A(1,qA_TM_M(zA_TM)).R3';refAbTM_TM_A(1,qA_TM_M(zA_TM)).R4'];
            eval(['CORRMAT_TM_A' num2str(zA_TM) ' =CMAT_TM_A']);
          %  newsum=newsum1+newsum;
            
            end
        
                    CORRMAT_TM_A=[CORRMAT_TM_A1 CORRMAT_TM_A2 CORRMAT_TM_A3 CORRMAT_TM_A4 CORRMAT_TM_A5];
           % newsum=newsum/5;
           % TMBlocksNew(1,r).B = newsum;
            TARTEMP_TM_A=[refLeTM_TM_A(1,rTMBB).C4;refLeTM_TM_A(1,rTMBB).C3;refLeTM_TM_A(1,rTMBB).C2;refLeTM_TM_A(1,rTMBB).C1;refAbTM_TM_A(1,rTMBB).R1';refAbTM_TM_A(1,rTMBB).R2';refAbTM_TM_A(1,rTMBB).R3';refAbTM_TM_A(1,rTMBB).R4'];
            beqA_TM=1;
            beqbA_TM=2;
            AeqA_TM=ones(1,5);
            lbA_TM=zeros(5,1);
            ubA_TM=ones(5,1);
            AA_TM=AeqA_TM;
            
            X_graph_TM = lsqlin(CORRMAT_TM_A,TARTEMP_TM_A,AA_TM,beqbA_TM,AeqA_TM,beqA_TM,lbA_TM, ubA_TM); % optimization funtion to calculate optimized weight
            %  X = lsqlin(abs(CORRMAT),abs(TARTEMP),A,beqb,Aeq,beq,0,Inf);
            PBlock_TM_A= PA_TM1*X_graph_TM(1)+PA_TM2*X_graph_TM(2)+PA_TM3*X_graph_TM(3)+PA_TM4*X_graph_TM(4)+PA_TM5*X_graph_TM(5);
           % eResBlock = newsum;
            eResBlock_graph_TM = PBlock_TM_A;  % predicted block using TM in graph domain
XMAT1A_TM=zeros(5,(rf/rb)*(cf/cb)-5);
XMATA_TM(:,rTMBB-5) = X_graph_TM;
XMATA_TM=XMAT1A_TM+XMATA_TM;

         end
        
end

rTMBB=rTMBB+1;
 %  end
end

TMres_graph=eResBlock_graph_TM-preBlock; %estimated/predicted residual block using TM in graph domain



for mTMB=5:12:(cb+4)*cBlock*rBlock
    refabrowpatch1TMB=row_patchFrameTM(1,mTMB:(mTMB+7));
    refabrowpatch2TMB=row_patchFrameTM(2,mTMB:(mTMB+7));
    refabrowpatch3TMB=row_patchFrameTM(3,mTMB:(mTMB+7));
    refabrowpatch4TMB=row_patchFrameTM(4,mTMB:(mTMB+7));
    refLecolpatch1TMB=row_patchFrameTM(1:12,mTMB-1);
    refLecolpatch2TMB=row_patchFrameTM(1:12,mTMB-2);
    refLecolpatch3TMB=row_patchFrameTM(1:12,mTMB-3);
    refLecolpatch4TMB=row_patchFrameTM(1:12,mTMB-4);
    row_patchFrameTMBBlock=row_patchFrameTM(5:12,mTMB:mTMB+7);
    refAbTMB(1,rTMB).R1 = refabrowpatch1TMB;
    refAbTMB(1,rTMB).R2 = refabrowpatch2TMB;
    refAbTMB(1,rTMB).R3 = refabrowpatch3TMB;
    refAbTMB(1,rTMB).R4 = refabrowpatch4TMB;
    refLeTMB(1,rTMB).C1 = refLecolpatch1TMB;
    refLeTMB(1,rTMB).C2 = refLecolpatch2TMB;
    refLeTMB(1,rTMB).C3 = refLecolpatch3TMB;
    refLeTMB(1,rTMB).C4 = refLecolpatch4TMB;
    TMBlocksTMB(1,rTMB).B = row_patchFrameTMBBlock;
    if rTMB<=5 && rBlock==1
      %     eResBlock = MY_estBlocks7(refAb, refLe, rb, cb, mode); % first 5 blocks of Predicted residual by PI
           eResBlockTMB = recons_block_final;% first 5 blocks of reconctruced pixels
           TMBBlocksNew(1,rTMB).B = eResBlockTMB;
    else
        if rTMB>=6
        
        for rTMB1=1:1:rTMB-1
        Abdiff1TMB=sum(imabsdiff(refAbTMB(1,rTMB).R1,refAbTMB(1,rTMB1).R1));
        Abdiff2TMB=sum(imabsdiff(refAbTMB(1,rTMB).R2,refAbTMB(1,rTMB1).R2));
        Abdiff3TMB=sum(imabsdiff(refAbTMB(1,rTMB).R3,refAbTMB(1,rTMB1).R3));
        Abdiff4TMB=sum(imabsdiff(refAbTMB(1,rTMB).R4,refAbTMB(1,rTMB1).R4));
        AbdiffTMB=Abdiff1TMB+Abdiff2TMB+Abdiff3TMB+Abdiff4TMB;
        Lediff1TMB=sum(imabsdiff(refLeTMB(1,rTMB).C1,refLeTMB(1,rTMB1).C1));
        Lediff2TMB=sum(imabsdiff(refLeTMB(1,rTMB).C2,refLeTMB(1,rTMB1).C2));
        Lediff3TMB=sum(imabsdiff(refLeTMB(1,rTMB).C3,refLeTMB(1,rTMB1).C3));
        Lediff4TMB=sum(imabsdiff(refLeTMB(1,rTMB).C4,refLeTMB(1,rTMB1).C4));
        LediffTMB=Lediff1TMB+Lediff2TMB+Lediff3TMB+Lediff4TMB;
        minAbdiffTMB(1,rTMB1)=AbdiffTMB;
        minLediffTMB(1,rTMB1)=LediffTMB;
        sumdiffTMB(1,rTMB1)=AbdiffTMB+LediffTMB;
        end        
        y1TMB=sumdiffTMB;
        nTMB=5;
        [ysTMB, indexTMB] = sort(y1TMB);
        qTMB=indexTMB(1:nTMB);
 
 
            for z1TMB=1:1:nTMB
            corrbloackTMB=TMBlocksTMB(1,qTMB(z1TMB)).B;
            eval(['PTMB' num2str(z1TMB) ' =corrbloackTMB']);
            CMATTMB=[refLeTMB(1,qTMB(z1TMB)).C4;refLeTMB(1,qTMB(z1TMB)).C3;refLeTMB(1,qTMB(z1TMB)).C2;refLeTMB(1,qTMB(z1TMB)).C1;refAbTMB(1,qTMB(z1TMB)).R1';refAbTMB(1,qTMB(z1TMB)).R2';refAbTMB(1,qTMB(z1TMB)).R3';refAbTMB(1,qTMB(z1TMB)).R4'];
            eval(['CORRMATTMB' num2str(z1TMB) ' =CMATTMB']);
          %  newsum=newsum1+newsum;
            
            end
            CORRMATTMB=[CORRMATTMB1 CORRMATTMB2 CORRMATTMB3 CORRMATTMB4 CORRMATTMB5];
           % newsum=newsum/5;
           % TMBlocksNew(1,r).B = newsum;
            TARTEMPTMB=[refLeTMB(1,rTMB).C4;refLeTMB(1,rTMB).C3;refLeTMB(1,rTMB).C2;refLeTMB(1,rTMB).C1;refAbTMB(1,rTMB).R1';refAbTMB(1,rTMB).R2';refAbTMB(1,rTMB).R3';refAbTMB(1,rTMB).R4'];
            beqTMB=1;
            beqbTMB=2;
            AeqTMB=ones(1,5);
            lbTMB=zeros(5,1);
            ubTMB=ones(5,1);
            ATMB=AeqTMB;
            
            XTMB = lsqlin(CORRMATTMB,TARTEMPTMB,ATMB,beqbTMB,AeqTMB,beqTMB,lbTMB, ubTMB); % optimization funtion to calculate optimized weight
            %  X = lsqlin(abs(CORRMAT),abs(TARTEMP),A,beqb,Aeq,beq,0,Inf);
            PBlockTMB= PTMB1*XTMB(1)+PTMB2*XTMB(2)+PTMB3*XTMB(3)+PTMB4*XTMB(4)+PTMB5*XTMB(5);
           % eResBlock = newsum;
            eResBlockTMB = PBlockTMB;  % estimated/predicted pixel block using TM B (pixel domain)
XMATTMB1=zeros(5,(rf/rb)*(cf/cb)-5);
XMATTMB(:,rTMB-5) = XTMB;
XMATTMB=XMATTMB1+XMATTMB;

        end
    end
    rTMB=rTMB+1;
end

TMBres=eResBlockTMB-preBlock; %estimated/predicted residual block using TM B

%%%%%%%%%------------ TEMPLATE MATCHING (pixel domain + graph domain) END--------------%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%------------ TEMPLATE POOLING (pixel domain + graph domain) START--------------%%%%%%%%%%%%%%%%%%%%



 [V_act,D_act] = eigenLap2(block);
 block_all=block(:);
  coeff_block= (V_act')*block_all;
  coeff_block_reshape=reshape(coeff_block,rb,cb);
  maxCoeff_block=max(max(abs(coeff_block)));
  minCoeff_block=min(min(abs(coeff_block))); 
  step=0.25;
  %step=1;
    for th=maxCoeff_block:-step:minCoeff_block-step
            
            mask=abs(coeff_block_reshape)>th;
            tmpBlock=mask.*coeff_block_reshape;
            eTmpBlock_block=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
             tmpblock_actual = reshape(tmpBlock,rb*cb,1);
                
            recons_block=(V_act)* tmpblock_actual;
            
           recons_block_final_NLM = reshape(recons_block,rb,cb); %Reconstructed block
    end
      %REFERENCE SAMPLES ABOVE for Actual Residual for TM TYPE B
     if rBlock==1
            refAbabTM=zeros(4,cb*3);
        else
            blockAb1abTM=reConsFrame_TM_B(((rBlock-2)*rb)+1:(rBlock-1)*rb,((cBlock-1)*cb)+1:cBlock*cb); %IMMEDIATELY ABOVE
            if cBlock>1 %ABOVE REFERENCE ARRAY CONTAINS THE CORNER REFERENCE
                blockAb2abTM=reConsFrame_TM_B(((rBlock-2)*rb)+1:(rBlock-1)*rb,((cBlock-2)*cb)+1:(cBlock-1)*cb); %ABOVE AND TO THE LEFT
            else
                blockAb2abTM(rb,1:cb)=blockAb1abTM(rb,1);
            end
            if cBlock<(cf/cb)
                blockAb3abTM=reConsFrame_TM_B(((rBlock-2)*rb)+1:(rBlock-1)*rb,((cBlock)*cb)+1:(cBlock+1)*cb); %ABOVE AND TO THE RIGHT
            else
                blockAb3abTM(rb,1:cb)=blockAb1abTM(rb,cb);
            end
          %  refAbab=[blockAb2ab(rb,:), blockAb1ab(rb,:), blockAb3ab(rb,:)];
          refAbabTM=[blockAb2abTM(5:rb,:), blockAb1abTM(5:rb,:), blockAb3abTM(5:rb,:)];
         end
          %REFERENCE SAMPLES TO THE LEFT for Actual residual for TM TYPE B
          
          if cBlock==1
          %  refLeLe=zeros((rb*2),2);
            refLeLeTM=zeros((rb*2),4);
        else
            blockLeLeTM=reConsFrame_TM_B(((rBlock-1)*rb)+1:(rBlock)*rb,((cBlock-2)*cb)+1:(cBlock-1)*cb); %IMMEDIATELY LEFT

              padLeTM=zeros(cb,4)+blockLeLeTM(rb,5:cb);
             refLeLeTM= [blockLeLeTM(:,5:cb); padLeTM];
        end
        

      refababTM= refAbab(:,cb+1:cb*2);
      expatchTM=[refababTM;recons_block_final_NLM];
      refableftTM=refAbabTM(:,cb-3:cb);
      refleaddTM=[refableftTM;refLeLeTM];
       reflesubTM=refleaddTM(1:12,:);
       
       patchfinalTM=[reflesubTM,expatchTM];
%onlyTP2_pixel=patchfinalTM;
      % AA=onlyTM2(5:12,5:12);
     %  onlyTM2_pixel(5:12,5:12)=0; 
  
         %CREATE  Patch FRAME for Reconstructed block
         rb_pTM=rb+4;
         cb_pTM=cb+4;
         for k=1:1:rb_pTM
             for j=1:1:cb_pTM
         
          patchFrameTM(((rBlock-1)*rb_p)+1:rBlock*rb_p,((cBlock-1)*cb_p)+1:cBlock*cb_p) = patchfinalTM;
             end
         end
         
        nTM=size(patchFrameTM,1);
 mTM=rb+4;
 row_patchFrameTM=[];
 norowsTM=nTM/mTM;
 
 for sTM= 1:norowsTM
row_patchFrameTM= [row_patchFrameTM,patchFrameTM(((sTM-1)*mTM+1):((sTM-1)*mTM+mTM),:)];
 end
 
 
       rnb=1;

for mTMB=5:12:(cb+4)*cBlock*rBlock
    refabrowpatch1TMB=row_patchFrameTM(1,mTMB:(mTMB+7));
    refabrowpatch2TMB=row_patchFrameTM(2,mTMB:(mTMB+7));
    refabrowpatch3TMB=row_patchFrameTM(3,mTMB:(mTMB+7));
    refabrowpatch4TMB=row_patchFrameTM(4,mTMB:(mTMB+7));
    refLecolpatch1TMB=row_patchFrameTM(1:12,mTMB-1);
    refLecolpatch2TMB=row_patchFrameTM(1:12,mTMB-2);
    refLecolpatch3TMB=row_patchFrameTM(1:12,mTMB-3);
    refLecolpatch4TMB=row_patchFrameTM(1:12,mTMB-4);
    row_patchFrameTMBBlock=row_patchFrameTM(5:12,mTMB:mTMB+7);
    refAbTMB(1,rnb).R1 = refabrowpatch1TMB;
    refAbTMB(1,rnb).R2 = refabrowpatch2TMB;
    refAbTMB(1,rnb).R3 = refabrowpatch3TMB;
    refAbTMB(1,rnb).R4 = refabrowpatch4TMB;
    refLeTMB(1,rnb).C1 = refLecolpatch1TMB;
    refLeTMB(1,rnb).C2 = refLecolpatch2TMB;
    refLeTMB(1,rnb).C3 = refLecolpatch3TMB;
    refLeTMB(1,rnb).C4 = refLecolpatch4TMB;
    TMBlocksNLMB(1,rnb).B = row_patchFrameTMBBlock;
    TMB_Col(1,rnb).TC = [refLeTM(1,rnb).C4;refLeTM(1,rnb).C3;refLeTM(1,rnb).C2;refLeTM(1,rnb).C1;refAbTM(1,rnb).R1';refAbTM(1,rnb).R2';refAbTM(1,rnb).R3';refAbTM(1,rnb).R4'];
    
    if rnb<=5 && rBlock==1
      
           %eResBlockTMB = recons_block_final;% first 5 blocks of reconctruced pixels
           PNLMBBlockFinal = recons_block_final_NLM; % estimated/predicted residual block using PI;
           TMBBlocksNew2(1,rnb).B = PNLMBBlockFinal;
    else
        if rnb>=6
        
        for rnb1=1:1:rnb-1
        NLMB(1,rnb1).SS =(imabsdiff(TM_Col(1,rnb).TC,TM_Col(1,rnb1).TC)).^2;
        NLMB(1,rnb1).SSADD = sum(NLMB(1,rnb1).SS);
        %NLM(1,rnb1).SSADD = sqrt(NLM(1,rnb1).SSADD);
        addB(rnb1)=NLMB(1,rnb1).SSADD; 
        end
        
        maxaddB=std(addB);
        SSADDMAXDIVB = addB/maxaddB;
        SSADDMAXDIVEXPB = exp(-SSADDMAXDIVB);
        SSADDMAXDIVEXPBW = SSADDMAXDIVEXPB/sum(SSADDMAXDIVEXPB);
        PNLMBBlockF=zeros(rb,cb);
        
        ub=rnb;
        for nbl=1:1:ub-1
        QB=TMBlocksNLMB(1,nbl).B;
        PNLMBBlock = QB*SSADDMAXDIVEXPBW(nbl);
        PNLMBBlockF = PNLMBBlockF + PNLMBBlock;
        PNLMBBlockFinal = PNLMBBlockF; %predicted block using Template pooling in pixel domain

        end
        
        end
    end
    rnb=rnb+1;
end

TMBresNLM=PNLMBBlockFinal-preBlock; %estimated/predicted residual block using TM B




%%%%%%%%%------------ TEMPLATE POOLING (pixel domain) END--------------%%%%%%%%%%%%%%%%%%%%


 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
   
         
        %COMPUTE ESTIMATED  RESIDUAL. THIS IS USEFUL SO THE DECODER CAN
        %RECONSTRUCT THE GRAPH FOR THE BLOCK.
        %THIS WHERE THE IDEAS OF THE PAPER ON INACCURACY MODELING ARR TO BE
        %APPLIED. 
%         eResBlock = estBlocks(refAb, refLe, rb, cb, mode);
%          eResBlock = My_estBlocks_Verti(refAb, refLe, rb, cb, mode);
        %  eResBlock = test1(refAb, refLe, rb, cb, mode);
%          eResBlock = new3(refAb, refLe, rb, cb, mode);
       % eResBlock = MY_estBlocks3(refAb, refLe, rb, cb, mode);
        %  eResBlock = MY_estBlocks5(refAb, refLe, rb, cb, mode);
          % eResBlock = MY_estBlocks6(refAb, refLe, rb, cb, mode);
          eResBlock_PI = MY_estBlocks7(refAb, refLe, rb, cb, mode); % estimated/predicted residual block using PI
        %  PNLMBlockFinal = MY_estBlocks7(refAb, refLe, rb, cb, mode); % estimated/predicted residual block using PI;
%{          
          
          
 if rBlock==1 && cBlock<=5         
  eResBlock = MY_estBlocks7(refAb, refLe, rb, cb, mode);
 else
  eResBlock = MY_estBlocks7_TM(rFrame, rb, cb, rBlock, cBlock);
 end       
 
 
 %}
     %   eResBlock_patch = MY_estBlocks7_tm(patch);
        %COMPUTE difference between actual residual and approximated residual
        
 %     DiffRes=abs(eResBlock-resBlock);
%        DiffRes_PI=abs(eResBlock_PI-resBlock); 
        %COMPUTE GRAPH REPRESENTATION USING ESTIMATED RESIDUAL
        [V_PI,D_PI] = eigenLap2(eResBlock_PI);
         [V_PI_2D,D_PI_2D] = eigenLap2_new_2D(eResBlock_PI);
       %   [V_PI_2D,D_PI_2D] = eigenLap2_new_exp(eResBlock_PI);
     %  [V_PI_2D,D_PI_2D] = eigenLap2_new_2D_back(eResBlock_PI);
        
        %[V,D] = eigenLapALL(eResBlock);
        [V,D] = eigenLap2(eResBlock);% template matching in residual domain with optimization
        [V_2D,D_2D] = eigenLap2_new_2D(eResBlock);% template matching in residual domain with optimization in 2D gbt self loop
      %[V_2D,D_2D] = eigenLap2_new_2D_back(eResBlock);
      %  [V_2D,D_2D] = eigenLap2_new_exp(eResBlock);
     % [V,D] = eig(eResBlock);
        
      [V_TMANLM,D_TMANLM] = eigenLap2(PNLMBlockFinal); % template matching in residual domain with weighted average
          [V_TMANLM_2D,D_TMANLM_2D] = eigenLap2_new_2D(PNLMBlockFinal); % template matching in residual domain with weighted average in 2D gbt self loop
      %  [V_TMANLM_2D,D_TMANLM_2D] = eigenLap2_new_2D_back(PNLMBlockFinal);
        % [V_TMANLM_2D,D_TMANLM_2D] = eigenLap2_new_exp(PNLMBlockFinal);
     
     
     
          
        [V_TMB,D_TMB] = eigenLap2(TMBres);% template matching in pixel domain with optimization
          [V_TMB_2D,D_TMB_2D] = eigenLap2_new_2D(TMBres);% template matching in pixel domain with optimization in 2D gbt self loop
     %  [V_TMB_2D,D_TMB_2D] = eigenLap2_new_2D_back(TMBres);
       %  [V_TMB_2D,D_TMB_2D] = eigenLap2_new_exp(TMBres);
          
         
       %  [V_TMBNLM,D_TMBNLM] = eigenLap2(TMBresNLM);  % template matching in pixel domain with weighted average
          [V_TMBNLM_2D_old,D_TMBNLM_2D_old] = eigenLap2_new_2D(TMBresNLM);  % template matching in pixel domain with weighted average in 2D gbt self loop
      %  [V_TMBNLM_2D,D_TMBNLM_2D] = eigenLap2_new_2D_back(TMBresNLM);
       %  [V_TMBNLM_2D,D_TMBNLM_2D] = eigenLap2_new_exp(TMBresNLM);
       %   [V_TMANLM,D_TMANLM] = eig(PNLMBlockFinal);
         
   %{     
        %COMPUTE GRAPH REPRESENTATION USING ESTIMATED RESIDUAL with 8
        %connectivity
        %[Ves,Des] = eigenLap2_test(eResBlock);
        [Ves,Des] = eigenLap2_8_connectivity(eResBlock);
   %}     
        %COMPUTE GRAPH REPRESENTATION USING ACTUAL RESIDUAL
        %[Vr,Dr] = eigenLapALL(resBlock);
       % [Vr,Dr] = eigenLap2(resBlock);
         [Vr,Dr] = eigenLap2_VLall_Self_lap(resBlock);
         [Vr_2D_old,Dr_2D_old] = eigenLap2_new_2D(resBlock);
         [Vr_2D,Dr_2D] = eigenLap2_new_2D_all(resBlock); % 2d self loop with unit edge 1 all connected
         
         
         
         [V_TMBNLM,D_TMBNLM] = eigenLap2_VLall_Self_lap(TMBresNLM);
         [V_TMBNLM_2D,D_TMB_2D] = eigenLap2_new_2D_all(TMBresNLM);% 
%  loss3_16_bestmode_self_256       
%%%%% Python Matrix start

%       load('loss3_16_bestmode1.mat','-ASCII')
      % load('loss3_16_bestmode1.mat')
       % load('data_loss3_1.mat')
        load('data_loss3_5_H_pred_UN_Kimono_L1small_dropout_1TT_Lrelu_20epo.mat','-ASCII') % dataset for actual residual
       %  load('data_loss3_5_H_pred.mat','-ASCII') % dataset for actual residual
       NNPredict = data_loss3_5_H_pred_UN_Kimono_L1small_dropout_1TT_Lrelu_20epo;
        NNPredict = reshape(data_loss3_5_H_pred_UN_Kimono_L1small_dropout_1TT_Lrelu_20epo,64,4096);
       NNW1=NNPredict(count,:);
%       eval(['NNW1P' num2str(count) ' =NNW1']);
%       rb=8;
%       cb=8;
       NNW=reshape(NNW1,64,64);
%       eval(['NNWPR' num2str(count) ' =NNW']);
       length1 = rb*cb;
       NNT=zeros(length1,length1);

       for jj=1:1:length1
       NNT(jj,jj)=sum(NNW(jj,:));
       end
%       eval(['NNTP' num2str(count) ' =NNT']);  
       NNL=NNT-NNW;
  %      NNL=NNW;
%       eval(['NNLP' num2str(count) ' =NNL']);
       [NNV,NND]=eig(NNL);
       NNV=real(NNV);
       NND=real(NND);
       eval(['NNVP' num2str(count) ' =NNV']);
%       eval(['NNDP' num2str(count) ' =NND']);
%%%%%       count=count+1;
       
       
       
        load('ipall_pre_model_320_64_bqmall_ClassC_best_64_pred.mat','-ASCII') % dataset for predicted residual
       NNPredict_pred = ipall_pre_model_320_64_bqmall_ClassC_best_64_pred;
       NNW1_pred=NNPredict_pred(count,:);
%       eval(['NNW1P' num2str(count) ' =NNW1']);
%       rb=8;
%       cb=8;
       NNW_pred=reshape(NNW1_pred,64,64);
%       eval(['NNWPR' num2str(count) ' =NNW']);
       length1 = rb*cb;
       NNT_pred=zeros(length1,length1);

       for jj=1:1:length1
       NNT_pred(jj,jj)=sum(NNW_pred(jj,:));
       end
%       eval(['NNTP' num2str(count) ' =NNT']);
       NNL_pred=NNT_pred-NNW_pred;
  %      NNL=NNW;
%       eval(['NNLP' num2str(count) ' =NNL']);
       [NNV_pred,NND_pred]=eig(NNL_pred);
       NNV_pred=real(NNV_pred);
       NND_pred=real(NND_pred);
       eval(['NNVP_pred' num2str(count) ' =NNV_pred']);
%       eval(['NNDP' num2str(count) ' =NND']);
     %  count=count+1;

%%%%% Python Matrix end 

%%%%%% Ortega methos starts

%load('D:\MATLAB\DCC2019finalcode\Graph_Learning-master\Graph_Learning-master\demohoriz.mat','-ASCII')
      %  load('D:\MATLAB\DCC2019finalcode\Graph_Learning-master\Graph_Learning-master\demohoriz.mat')
        load('D:\MATLAB\DCC2019finalcode\Graph_Learning-master\Graph_Learning-master\BlowingBubbles_lap_all_small.mat')
        ORT_Lap=Laplacian;
        ORT_Lap_pred1=ORT_Lap(count,:);
        ORT_Lap_pred=reshape(ORT_Lap_pred1,64,64);
        [ORTV,ORTD]=eig(ORT_Lap_pred);
        eval(['ORTVP' num2str(count) ' =ORTV']);
%        count=count+1;
%%%%%% Ortega methos ends


%%%%%% GBP ONL Lap starts


%load('D:\MATLAB\DCC2019finalcode\Graph_Learning-master\Graph_Learning-master\demohoriz.mat','-ASCII')
      %  load('D:\MATLAB\DCC2019finalcode\Graph_Learning-master\Graph_Learning-master\demohoriz.mat')
       % load('D:\PythonJupyter\ILearnDeepLearning.py-master\01_mysteries_of_neural_networks\03_numpy_neural_net\Predicted_data_Lap3.mat', '-ASCII')
        load('D:\PythonJupyter\ILearnDeepLearning.py-master\01_mysteries_of_neural_networks\03_numpy_neural_net\Predicted_data_Lap_Cactus.mat', '-ASCII')
        ONL_Lap=Predicted_data_Lap_Cactus';
        ONL_Lap_pred1=ONL_Lap(count,:);
        ONL_Lap_pred=reshape(ONL_Lap_pred1,64,64);
        [ONLV,ONLD]=eig(ONL_Lap_pred);
        
        Avg_Ref_New_R=Avg_Ref_New(:);
       % ONLV_R=(ONLV')*Avg_Ref_New_R;
          ONLV_R=(ONLV')*resBlock(:);
 %       eval(['ORTVP' num2str(count) ' =ORTV']);
        count=count+1;
        
%%%%%% GBP ONL Lap ends

         
        % [VL] = eigenLap2_L(resBlock);
         [VL,WL] = eigenLap2_W(resBlock);
         VL_F=VL(:);
         WL_R=WL(:);
         %res_WL_Final(rBlock*cBlock,:)=WL_R;
         res_WL_Final(aa+1,:)=WL_R;
        % [VLadj] = eigenLap2_VLadj(VL);
         [VLadj] = eigenLap2_VLadj(resBlock);
         VLadj_R=VLadj(:);
       %  res_VLadj_Final(rBlock*cBlock,:)=VLadj_R;
         res_VLadj_Final(aa+1,:)=VLadj_R;
         
         [VLall] = eigenLap2_VLall(resBlock);
         VLall_R=VLall(:);
         %res_VLall_Final(rBlock*cBlock,:)=VLall_R;
         res_VLall_Final(aa+1,:)=VLall_R;
         
         VLall_R_temp1=VLall_R;
         eval(['VLall_R_temp' num2str(aa+1) ' =VLall_R_temp1']);
         %aa=aa+1;
         %[Sadj] = eigenLap2_Sadj(S);
         %[Vr_2D,Dr_2D] = eigenLap2_new_exp(resBlock);
        
    %{    
        %COMPUTE GRAPH REPRESENTATION USING ACTUAL RESIDUAL with 8 connectivity
        %[Vac,Dac] = eigenLap2_test(resBlock);
        [Vac,Dac] = eigenLap2_8_connectivity(resBlock);
        
     %}   
         %CREATE  PREDICTED FRAME
        pFrame(((rBlock-1)*rb)+1:rBlock*rb,((cBlock-1)*cb)+1:cBlock*cb) = preBlock;
        modeFrame(rBlock,cBlock)=mode;
        
        %CREATE  ACTUAL RESIDUAL FRAME
        rFrame(((rBlock-1)*rb)+1:rBlock*rb,((cBlock-1)*cb)+1:cBlock*cb) = resBlock;
        modeFrame(rBlock,cBlock)=mode;
        
        %CREATE  PREDICTED RESIDUAL FRAME for Template Matching TYPE A
        prFrame_TM(((rBlock-1)*rb)+1:rBlock*rb,((cBlock-1)*cb)+1:cBlock*cb) = eResBlock;
        modeFrame(rBlock,cBlock)=mode;
        
         %CREATE  PREDICTED RESIDUAL FRAME for Template Matching TYPE A with NLM
        prFrame_TMA_NLM(((rBlock-1)*rb)+1:rBlock*rb,((cBlock-1)*cb)+1:cBlock*cb) = PNLMBlockFinal;
        modeFrame(rBlock,cBlock)=mode;
      
         
         %CREATE  PREDICTED RESIDUAL FRAME for Prediction Inaccuracy
        prFrame_PI(((rBlock-1)*rb)+1:rBlock*rb,((cBlock-1)*cb)+1:cBlock*cb) = eResBlock_PI;
        modeFrame(rBlock,cBlock)=mode;
          %CREATE   Reconstructed FRAME for Template Matching TYPE B
         reConsFrame_TM_B(((rBlock-1)*rb)+1:rBlock*rb,((cBlock-1)*cb)+1:cBlock*cb) = recons_block_final;
        modeFrame(rBlock,cBlock)=mode;
          %CREATE   Reconstructed FRAME for Template Matching TYPE B NLM
         reConsFrame_TM_BNLM(((rBlock-1)*rb)+1:rBlock*rb,((cBlock-1)*cb)+1:cBlock*cb) = recons_block_final_NLM;
        modeFrame(rBlock,cBlock)=mode;
         %CREATE  PREDICTED RESIDUAL FRAME for Template Matching TYPE B
        prFrame_TM_B(((rBlock-1)*rb)+1:rBlock*rb,((cBlock-1)*cb)+1:cBlock*cb) = TMBres;
        modeFrame(rBlock,cBlock)=mode;
        
        %CREATE  PREDICTED RESIDUAL FRAME for Template Matching TYPE B
        prFrame_TM_BNLM(((rBlock-1)*rb)+1:rBlock*rb,((cBlock-1)*cb)+1:cBlock*cb) = TMBresNLM;
        modeFrame(rBlock,cBlock)=mode;
        
        %Reconstructed Pixel frame by template matching TYPE B with NLM
        Recons_pixel_TMBNLM(((rBlock-1)*rb)+1:rBlock*rb,((cBlock-1)*cb)+1:cBlock*cb) = PNLMBBlockFinal;
        modeFrame(rBlock,cBlock)=mode;
        
       % Reconstructed Pixel frame by template matching
        Recons_pixel_TMB(((rBlock-1)*rb)+1:rBlock*rb,((cBlock-1)*cb)+1:cBlock*cb) = eResBlockTMB;
        modeFrame(rBlock,cBlock)=mode;
        
        
       
      %  patchFrame(((rBlock-1)*rb)+1:rBlock*rb,((cBlock-1)*cb)+1:cBlock*cb) = patch;
      
   
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
        %COMPUTE DIFFERENT TRANSFORMS FOR EACH BLOCK
  %}    
     %DCT
        DCTres = dct2(resBlock);
        
        
        %ASI - THIS IS A TYPE OF SINE TRANSFORM
        N2=8; NN=N2;
        kk=0:N2-1;  ll=0:N2-1;
        DST7=sin(pi/(NN+1/2)*(kk+1/2)'*(ll+1));
        
        %DST7=round(DST7*(2/sqrt((2*N2)+1))*128);
        DST7=DST7*(2/sqrt((2*N2)+1));
        
        %DST1=sin(pi/(NN+1)*(kk+1)'*(ll+1));
        %DST1_coeff= resblock(col)*DST1;
        
        
        if (mode>=0 && mode<=10) || (mode>=26 && mode<=34)
            
            ASIrest = dct(resBlock');%thDSe DCT is applied to each row
            ASIrest=ASIrest';
            
            for col=1:1:cb
                ASIres(:,col)=ASIrest(:,col)'*DST7; %the DST is applied to each col
            end
        end
        
        if mode>=11 && mode<=25

            for col=1:1:cb
                ASIrest(:,col)=resBlock(:,col)'*DST7;
            end
            
            for row=1:1:rb
                ASIres(row,:)=ASIrest(row,:)*DST7;
            end
        end
        
        %KLT
        mn=mean(mean(resBlock));
        resBlock2=resBlock-mn;
        %resBlock3=reshape(resBlock2, cb*rb,1);
        resBlock3=resBlock2(:);
        R = (resBlock3*resBlock3')/((cb*rb)-1);
        [vvv,ddd]=eig(R);
        %KLTres=(vvv')*(reshape(resBlock, cb*rb,1));
        KLTres=(vvv')*resBlock(:);
        KLTres=reshape(KLTres,rb,cb);
        
%        diagtest0=diag(R);
        diagtest1=diag(R)+0.5;
        diagtest2=diag(diagtest1);
        diagtest_final=(R+diagtest2)-diag(diag(R));
        
%        R_INV=inv(R);
        R_INV=inv(diagtest_final);
        Re_R_INV = reshape(R_INV,1,64*64);
        res_Cov_INV_Final(aa+1,:)=Re_R_INV;
        
       % R_diag_INV = diag((1 ./ diag(R)));
        R_diag_INV = diag((1 ./ diag(diagtest_final)));
        Re_R_diag_INV = reshape(R_diag_INV,1,64*64);
        res_Cov_diag_INV_Final(aa+1,:)=Re_R_diag_INV;
        
        Re_R = reshape(R,1,64*64);
        %res_Cov_Final(rBlock*cBlock,:)=Re_R;
        res_Cov_Final(aa+1,:)=Re_R;
        %GRAPH-BASED USING APPROXIMATED RESIDUAL Template matching type A
        epsilon1=1E-10;
        [Adj3dgt] = test_2D_all(resBlock);
        Adj3dgt_temp=abs(Adj3dgt);
        Bmax = max(Adj3dgt_temp(:));
        Bmin = min(Adj3dgt_temp(:));
        RangeB = Bmax - Bmin;
        Bnrm = ((Adj3dgt_temp - Bmin)/(RangeB + epsilon1));
       % eval(['Adj3dgt' num2str(count-1) ' =Adj3dgt']);
        eval(['Adj3dgt' num2str(count-1) ' =Bnrm']);
        %Block_Adj3dgt_3D=cat(3,Adj3dgt1,Adj3dgt2,Adj3dgt3,Adj3dg4);
                
        %Concatenate residual values into a N^2X1 vector
        x = resBlock(:);
        %for c=2:1:cb
           % x=[x; resBlock(:,c)];
        %end
        
        yt=(V')*x;
        
        y=reshape(yt,rb,cb); %These are the coeffcients obtained by using graph-based transform by template matching
        
        if abs(sum(sum(KLTres.^2)) - sum(sum(y.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (approx.) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             ye=sum(sum(y.^2));
        end
        
         %Concatenate residual values into a N^2X1 vector 
        x = resBlock(:);
        %for c=2:1:cb
           % x=[x; resBlock(:,c)];
        %end
        
        yt_ORT=(ORTV')*x;
        
        y_ORT=reshape(yt_ORT,rb,cb); %These are the coeffcients obtained by using Ortega's method
        
        if abs(sum(sum(KLTres.^2)) - sum(sum(y_ORT.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (approx.) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             ye_ORT=sum(sum(y_ORT.^2));
        end
        
        %GRAPH-BASED USING APPROXIMATED RESIDUAL Template matching type A with 2D GBT self loop
        %with 2D GBT self loop
        
        %Concatenate residual values into a N^2X1 vector
        x = resBlock(:);
        %for c=2:1:cb
           % x=[x; resBlock(:,c)];
        %end
        
        yt_2D=(V_2D')*x;
        
        y_2D=reshape(yt_2D,rb,cb); %These are the coeffcients obtained by using graph-based transform by template matching
        
        if abs(sum(sum(KLTres.^2)) - sum(sum(y_2D.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (approx.) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             ye_2D=sum(sum(y_2D.^2));
        end
        
         %GRAPH-BASED USING APPROXIMATED RESIDUAL Template matching type A NLM
         %NLM
        
        %Concatenate residual values into a N^2X1 vector
        x = resBlock(:);
        %for c=2:1:cb
           % x=[x; resBlock(:,c)];
        %end
        
        ytnlm=(V_TMANLM')*x;
        
        ynlm=reshape(ytnlm,rb,cb); %These are the coeffcients obtained by using graph-based transform by template matching
        
        if abs(sum(sum(KLTres.^2)) - sum(sum(ynlm.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (approx.) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             yenlm=sum(sum(ynlm.^2));
        end
        
      %GRAPH-BASED USING APPROXIMATED RESIDUAL Template matching type A NLM with 2D GBT self loop
         %NLM
        
        %Concatenate residual values into a N^2X1 vector
        x = resBlock(:);
        %for c=2:1:cb
           % x=[x; resBlock(:,c)];
        %end
        
        ytnlm_2D=(V_TMANLM_2D')*x;
        
        ynlm_2D=reshape(ytnlm_2D,rb,cb); %These are the coeffcients obtained by using graph-based transform by template matching
        
        if abs(sum(sum(KLTres.^2)) - sum(sum(ynlm_2D.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (approx.) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             yenlm_2D=sum(sum(ynlm_2D.^2));
        end
            
        
        ONLV_R_R=reshape(ONLV_R,rb,cb); %GBT-ONL cofficient
        
        if abs(sum(sum(KLTres.^2)) - sum(sum(ONLV_R_R.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT ONL (approx.) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             ONLV_R_R_S=sum(sum(ONLV_R_R.^2));
        end
        
        
        
                %%% GRAPH-BASED USING APPROXIMATED RESIDUAL Template matching type A separable %%%%

if mode < 18
    bb=resBlock;  %% Horizontal Block
    bberes=eResBlock; %% Horizontal Block
    fprintf('TM Type A Horizonal mode = %i\n', mode)
else
    bb=resBlock'; %% Vertical Block
    bberes=eResBlock'; %% Vertical Block
    fprintf('TM Type A Vertical mode = %i\n', mode)
end

rberes=size(bberes,1);
cberes=size(bberes,2);
for r=1:1:rberes
     ca=bberes(r,:);
     c=bb(r,:);
     %eval(['ROW' num2str(r) ' =c']);
     catemp = ca;
     ctemp = c;
     %eval(['D' num2str(r) ' =d']);
epsilon=1E-10;
CN=abs(catemp);
%NBabs=(CN - max(CN)) / (min(CN) - max(CN)+epsilon); % backward normalisation
NFabs=(CN - min(CN)) / (max(CN) - min(CN)+epsilon);  % Forward normalisation
for i=1:1:cb
    if NFabs(1,i)>=1
       NFabs(1,i)=1;
    end
end

% GBT Calculation
W=zeros(rb,cb);
for c1=1:1:cb-1
        W(c1,(c1+1))=1;
        W((c1+1),c1)=1;  
end
%{
WDabs=zeros(rb,cb);
for i=1:1:cb
    WDabs(i,i)=NFabs(i);
    if NFabs(i)==0
       WDabs(i,i)=epsilon;
    end
end
%}
WFabs=W;

T=zeros(rb,cb);
for j=1:1:8
    T(j,j)=ceil(sum(WFabs(j,:)));
end
%{
TDabs=zeros(rb,cb);
for j=1:1:8
    TDabs(j,j)=ceil(sum(WDabs(j,:)));
end
%}
TFabs=T;
LFabs=TFabs-WFabs;
[VFabs,DFabs]=eig(LFabs);
eval(['V1TMA' num2str(r) ' =VFabs']);
TMFA= (VFabs')*(ctemp');
TMFAT=TMFA';
TMFAtemp=zeros(rb,cb);
TMFFA(r,:) = TMFAT;
TMFFA=TMFAtemp+TMFFA;
end


if mode < 18
    TMFFA=TMFFA;
else
    TMFFA=TMFFA';
end

if mode < 18

%%%% HORIZONTAL-Veritical Calculation Start %%%%

%fprintf('Horizonal mode = %i\n', mode)

% GBT Original Block Vector Calculation Start
TMFFAT=TMFFA';
vTMFFAT=bberes';
rberes=size(vTMFFAT,1);
cberes=size(vTMFFAT,2);
for r=1:1:rberes
     vda=vTMFFAT(r,:);
     da=TMFFAT(r,:);
     %eval(['COL' num2str(r) ' =d']);
     vdatemp = vda;
     datemp = da;
     %eval(['D' num2str(r) ' =d']);
epsilon=1E-10;
DN=abs(vdatemp);
%NBabs=(CN - max(CN)) / (min(CN) - max(CN)+epsilon); % backward normalisation
NDFabs=(DN - min(DN)) / (max(DN) - min(DN)+epsilon);  % Forward normalisation
for i=1:1:cb
    if NDFabs(1,i)>=1
       NDFabs(1,i)=1;
    end
end

W1=zeros(rb,cb);
for c1=1:1:cb-1
        W1(c1,(c1+1))=1;
        W1((c1+1),c1)=1;  
end

W1Dabs=zeros(rb,cb);
for i=1:1:cb
    W1Dabs(i,i)=NDFabs(i);
    if NDFabs(i)==0
       W1Dabs(i,i)=epsilon;
    end
end

W1Fabs=W1+W1Dabs;

T1=zeros(rb,cb);

for j=1:1:8
    T1(j,j)=ceil(sum(W1Fabs(j,:)));
end

T1Dabs=zeros(rb,cb);
for j=1:1:8
    T1Dabs(j,j)=ceil(sum(W1Dabs(j,:)));
end

T1Fabs=T1+T1Dabs;
L1Fabs=T1Fabs-W1Fabs;
[V1Fabs,D1Fabs]=eig(L1Fabs);
eval(['V2TMA' num2str(r) ' =V1Fabs']);
TMH= (V1Fabs')*(datemp');
TMHT=TMH';
TMHH=zeros(rb,cb);
TMFinaleres(r,:) = TMHT;
TMFinaleres=TMHH+TMFinaleres;
end
TMFinaleres=TMFinaleres';


%%%% HORIZONTAL-Veritical Calculation End %%%%


else

%fprintf('Verical mode = %i\n', mode)

%%%% VERTICAL-Horizontal Calculation Start %%%%

% GBT Original Block Vector Calculation Start

vTMFFA=bberes;
rberes=size(vTMFFA,1);
cberes=size(vTMFFA,2);
for r=1:1:rberes
     vda=vTMFFA(r,:);
     da=TMFFA(r,:);
     %eval(['COL' num2str(r) ' =d']);
     vdatemp = vda;
     datemp = da;
     %eval(['D' num2str(r) ' =d']);
epsilon=1E-10;

DN=abs(vdatemp);
%NBabs=(CN - max(CN)) / (min(CN) - max(CN)+epsilon); % backward normalisation
NDFabs=(DN - min(DN)) / (max(DN) - min(DN)+epsilon);  % Forward normalisation
for i=1:1:cb
    if NDFabs(1,i)>=1
       NDFabs(1,i)=1;
    end
end

W1=zeros(rb,cb);
for c1=1:1:cb-1
        W1(c1,(c1+1))=1;
        W1((c1+1),c1)=1;  
end

W1Dabs=zeros(rb,cb);
for i=1:1:cb
    W1Dabs(i,i)=NDFabs(i);
    if NDFabs(i)==0
       W1Dabs(i,i)=epsilon;
    end
end

W1Fabs=W1+W1Dabs;

T1=zeros(rb,cb);
for j=1:1:8
    T1(j,j)=ceil(sum(W1Fabs(j,:)));
end

T1Dabs=zeros(rb,cb);
for j=1:1:8
    T1Dabs(j,j)=ceil(sum(W1Dabs(j,:)));
end

T1Fabs=T1+T1Dabs;
L1Fabs=T1Fabs-W1Fabs;
[V1Fabs,D1Fabs]=eig(L1Fabs);
eval(['V2TMA' num2str(r) ' =V1Fabs']);
TMH= (V1Fabs')*(datemp');
TMHT=TMH';
TMHH=zeros(rb,cb);
TMFinaleres(r,:) = TMHT;
TMFinaleres=TMHH+TMFinaleres;
end


% GBT Original Block Vector Calculation End

%%%% VERTICAL-Horizontal Calculation End %%%%

end
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
         %GRAPH-BASED USING APPROXIMATED RESIDUAL Template matching TYPE B with optimization
         
        
        %Concatenate residual values into a N^2X1 vector
        x = resBlock(:);
        %for c=2:1:cb
           % x=[x; resBlock(:,c)];
        %end
        
        yt_TMB=(V_TMB')*x;
        
        y_TMB=reshape(yt_TMB,rb,cb); %These are the coeffcients obtained by using graph-based transform by template matching TYPE B
        
        if abs(sum(sum(KLTres.^2)) - sum(sum(y_TMB.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (approx.) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             ye_TMB=sum(sum(y_TMB.^2));
        end
          %GRAPH-BASED USING APPROXIMATED RESIDUAL Template matching TYPE B with optimization in 2D GBT with self loop
          
        %Concatenate residual values into a N^2X1 vector
        x = resBlock(:);
        %for c=2:1:cb
           % x=[x; resBlock(:,c)];
        %end
        
        yt_TMB_2D=(V_TMB_2D')*x;
        
        y_TMB_2D=reshape(yt_TMB_2D,rb,cb); %These are the coeffcients obtained by using graph-based transform by template matching TYPE B
        
        if abs(sum(sum(KLTres.^2)) - sum(sum(y_TMB_2D.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (approx.) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             ye_TMB_2D=sum(sum(y_TMB_2D.^2));
        end     
        
        
         %GRAPH-BASED USING APPROXIMATED RESIDUAL Template matching TYPE B NLM
         %NLM
        
        %Concatenate residual values into a N^2X1 vector
        x = resBlock(:);
        %for c=2:1:cb
           % x=[x; resBlock(:,c)];
        %end
        
        yt_TMBnlm=(V_TMBNLM')*x;
        
        y_TMBnlm=reshape(yt_TMBnlm,rb,cb); %These are the coeffcients obtained by using graph-based transform by template matching TYPE B
        
        if abs(sum(sum(KLTres.^2)) - sum(sum(y_TMBnlm.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (approx.) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             ye_TMBnlm=sum(sum(y_TMBnlm.^2));
        end        
        
        
       %GRAPH-BASED USING APPROXIMATED RESIDUAL Template matching TYPE B NLM in 2D GBT with self loop
         %NLM
        
        %Concatenate residual values into a N^2X1 vector
        x = resBlock(:);
        %for c=2:1:cb
           % x=[x; resBlock(:,c)];
        %end
        
        yt_TMBnlm_2D=(V_TMBNLM_2D')*x;
        
        y_TMBnlm_2D=reshape(yt_TMBnlm_2D,rb,cb); %These are the coeffcients obtained by using graph-based transform by template matching TYPE B
        
        if abs(sum(sum(KLTres.^2)) - sum(sum(y_TMBnlm_2D.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (approx.) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             ye_TMBnlm_2D=sum(sum(y_TMBnlm_2D.^2));
        end          
        
         yt_TMBnlm_2D=(V_TMBNLM_2D')*x;
        
        y_TMBnlm_2D=reshape(yt_TMBnlm_2D,rb,cb); %These are the coeffcients obtained by using graph-based transform by template matching TYPE B
        
        if abs(sum(sum(KLTres.^2)) - sum(sum(y_TMBnlm_2D.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (approx.) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             ye_TMBnlm_2D=sum(sum(y_TMBnlm_2D.^2));
        end 
        
       
         x = resBlock(:);
        %for c=2:1:cb
           % x=[x; resBlock(:,c)];
        %end
        
        yt_TMBnlm_2D_old=(V_TMBNLM_2D_old')*x;
        
        y_TMBnlm_2D_old=reshape(yt_TMBnlm_2D_old,rb,cb); %These are the coeffcients obtained by using graph-based transform by template matching TYPE B
        
        if abs(sum(sum(KLTres.^2)) - sum(sum(y_TMBnlm_2D_old.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (approx.) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             ye_TMBnlm_2D=sum(sum(y_TMBnlm_2D_old.^2));
        end          
        
         yt_TMBnlm_2D_old=(V_TMBNLM_2D_old')*x;
        
        y_TMBnlm_2D_old=reshape(yt_TMBnlm_2D_old,rb,cb); %These are the coeffcients obtained by using graph-based transform by template matching TYPE B
        
        if abs(sum(sum(KLTres.^2)) - sum(sum(y_TMBnlm_2D_old.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (approx.) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             ye_TMBnlm_2D_old=sum(sum(y_TMBnlm_2D_old.^2));
        end 
        
        
        
        
                %%% GRAPH-BASED USING APPROXIMATED RESIDUAL Template matching type B separable %%%%

if mode < 18
    bb=resBlock;  %% Horizontal Block
    bberes=TMBres; %% Horizontal Block
    fprintf('TM Type B Horizonal mode = %i\n', mode)
else
    bb=resBlock'; %% Vertical Block
    bberes=TMBres'; %% Vertical Block
    fprintf('TM Type B Vertical mode = %i\n', mode)
end

rberes=size(bberes,1);
cberes=size(bberes,2);
for r=1:1:rberes
     ca=bberes(r,:);
     c=bb(r,:);
     %eval(['ROW' num2str(r) ' =c']);
     catemp = ca;
     ctemp = c;
     %eval(['D' num2str(r) ' =d']);
epsilon=1E-10;
CN=abs(catemp);
%NBabs=(CN - max(CN)) / (min(CN) - max(CN)+epsilon); % backward normalisation
NFabs=(CN - min(CN)) / (max(CN) - min(CN)+epsilon);  % Forward normalisation
for i=1:1:cb
    if NFabs(1,i)>=1
       NFabs(1,i)=1;
    end
end

% GBT Calculation
W=zeros(rb,cb);
for c1=1:1:cb-1
        W(c1,(c1+1))=1;
        W((c1+1),c1)=1;  
end
%{
WDabs=zeros(rb,cb);
for i=1:1:cb
    WDabs(i,i)=NFabs(i);
    if NFabs(i)==0
       WDabs(i,i)=epsilon;
    end
end
%}
WFabs=W;

T=zeros(rb,cb);
for j=1:1:8
    T(j,j)=ceil(sum(WFabs(j,:)));
end
%{
TDabs=zeros(rb,cb);
for j=1:1:8
    TDabs(j,j)=ceil(sum(WDabs(j,:)));
end
%}
TFabs=T;
LFabs=TFabs-WFabs;
[VFabs,DFabs]=eig(LFabs);
eval(['V1TMB' num2str(r) ' =VFabs']);
TMFB= (VFabs')*(ctemp');
TMFBT=TMFB';
TMFBtemp=zeros(rb,cb);
TMFFB(r,:) = TMFBT;
TMFFB=TMFBtemp+TMFFB;
end


if mode < 18
    TMFFB=TMFFB;
else
    TMFFB=TMFFB';
end

if mode < 18

%%%% HORIZONTAL-Veritical Calculation Start %%%%

%fprintf('Horizonal mode = %i\n', mode)

% GBT Original Block Vector Calculation Start
TMFFBT=TMFFB';
vTMFFBT=bberes';
rberes=size(vTMFFBT,1);
cberes=size(vTMFFBT,2);
for r=1:1:rberes
     vda=vTMFFBT(r,:);
     da=TMFFBT(r,:);
     %eval(['COL' num2str(r) ' =d']);
     vdatemp = vda;
     datemp = da;
     %eval(['D' num2str(r) ' =d']);
epsilon=1E-10;
DN=abs(vdatemp);
%NBabs=(CN - max(CN)) / (min(CN) - max(CN)+epsilon); % backward normalisation
NDFabs=(DN - min(DN)) / (max(DN) - min(DN)+epsilon);  % Forward normalisation
for i=1:1:cb
    if NDFabs(1,i)>=1
       NDFabs(1,i)=1;
    end
end

W1=zeros(rb,cb);
for c1=1:1:cb-1
        W1(c1,(c1+1))=1;
        W1((c1+1),c1)=1;  
end

W1Dabs=zeros(rb,cb);
for i=1:1:cb
    W1Dabs(i,i)=NDFabs(i);
    if NDFabs(i)==0
       W1Dabs(i,i)=epsilon;
    end
end

W1Fabs=W1+W1Dabs;

T1=zeros(rb,cb);

for j=1:1:8
    T1(j,j)=ceil(sum(W1Fabs(j,:)));
end

T1Dabs=zeros(rb,cb);
for j=1:1:8
    T1Dabs(j,j)=ceil(sum(W1Dabs(j,:)));
end

T1Fabs=T1+T1Dabs;
L1Fabs=T1Fabs-W1Fabs;
[V1Fabs,D1Fabs]=eig(L1Fabs);
eval(['V2TMB' num2str(r) ' =V1Fabs']);
TMBH= (V1Fabs')*(datemp');
TMBHT=TMBH';
TMBHH=zeros(rb,cb);
TMBFinaleres(r,:) = TMBHT;
TMBFinaleres=TMBHH+TMBFinaleres;
end
TMBFinaleres=TMBFinaleres';


%%%% HORIZONTAL-Veritical Calculation End %%%%


else

%fprintf('Verical mode = %i\n', mode)

%%%% VERTICAL-Horizontal Calculation Start %%%%

% GBT Original Block Vector Calculation Start

vTMBFFA=bberes;
rberes=size(vTMBFFA,1);
cberes=size(vTMBFFA,2);
for r=1:1:rberes
     vda=vTMBFFA(r,:);
     da=TMFFB(r,:);
     %eval(['COL' num2str(r) ' =d']);
     vdatemp = vda;
     datemp = da;
     %eval(['D' num2str(r) ' =d']);
epsilon=1E-10;

DN=abs(vdatemp);
%NBabs=(CN - max(CN)) / (min(CN) - max(CN)+epsilon); % backward normalisation
NDFabs=(DN - min(DN)) / (max(DN) - min(DN)+epsilon);  % Forward normalisation
for i=1:1:cb
    if NDFabs(1,i)>=1
       NDFabs(1,i)=1;
    end
end

W1=zeros(rb,cb);
for c1=1:1:cb-1
        W1(c1,(c1+1))=1;
        W1((c1+1),c1)=1;  
end

W1Dabs=zeros(rb,cb);
for i=1:1:cb
    W1Dabs(i,i)=NDFabs(i);
    if NDFabs(i)==0
       W1Dabs(i,i)=epsilon;
    end
end

W1Fabs=W1+W1Dabs;

T1=zeros(rb,cb);
for j=1:1:8
    T1(j,j)=ceil(sum(W1Fabs(j,:)));
end

T1Dabs=zeros(rb,cb);
for j=1:1:8
    T1Dabs(j,j)=ceil(sum(W1Dabs(j,:)));
end

T1Fabs=T1+T1Dabs;
L1Fabs=T1Fabs-W1Fabs;
[V1Fabs,D1Fabs]=eig(L1Fabs);
eval(['V2TMB' num2str(r) ' =V1Fabs']);
TMBH= (V1Fabs')*(datemp');
TMBHT=TMBH';
TMBHH=zeros(rb,cb);
TMBFinaleres(r,:) = TMBHT;
TMBFinaleres=TMBHH+TMBFinaleres;
end


% GBT Original Block Vector Calculation End

%%%% VERTICAL-Horizontal Calculation End %%%%

end       
        
        
        
        
        
        
        
        
        
        
        
   %{     
         %GRAPH-BASED USING APPROXIMATED RESIDUAL with 8 connectivity
                
        yte=(Ves')*x;
        
        ye=reshape(yte,rb,cb); %These are the coeffcients obtained by using graph-based transform with 8 connectivity
        
        if abs(sum(sum(KLTres.^2)) - sum(sum(ye.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (approx.) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             yes=sum(sum(ye.^2));
        end
   %}    
        
     %GRAPH-BASED USING APPROXIMATED RESIDUAL Prediction Inaccuracy
        
        %Concatenate residual values into a N^2X1 vector
        x = resBlock(:);
        %for c=2:1:cb
           % x=[x; resBlock(:,c)];
        %end
        
        yt_PI=(V_PI')*x;
        
        y_PI=reshape(yt_PI,rb,cb); %These are the coeffcients obtained by using graph-based transform
        
        if abs(sum(sum(KLTres.^2)) - sum(sum(y_PI.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (approx.) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             ye_PI=sum(sum(y_PI.^2));
        end
        
    %GRAPH-BASED USING APPROXIMATED RESIDUAL Prediction Inaccuracy with 2D GBT self loop
   
        
        %Concatenate residual values into a N^2X1 vector
        x = resBlock(:);
        %for c=2:1:cb
           % x=[x; resBlock(:,c)];
        %end
        
        yt_PI_2D=(V_PI_2D')*x;
        
        y_PI_2D=reshape(yt_PI_2D,rb,cb); %These are the coeffcients obtained by using graph-based transform
        
        if abs(sum(sum(KLTres.^2)) - sum(sum(y_PI_2D.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (approx.) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             ye_PI_2D=sum(sum(y_PI_2D.^2));
        end     
        
        
        
        
        
 %%% GBT Approximate PI separable %%%%

if mode < 18
    bb=resBlock;  %% Horizontal Block
    bberes=eResBlock_PI; %% Horizontal Block
    fprintf('GBT Approximation Horizonal mode = %i\n', mode)
else
    bb=resBlock'; %% Vertical Block
    bberes=eResBlock_PI'; %% Vertical Block
    fprintf('GBT Approximation Vertical mode = %i\n', mode)
end

rberes=size(bberes,1);
cberes=size(bberes,2);
for r=1:1:rberes
     ca=bberes(r,:);
     c=bb(r,:);
     %eval(['ROW' num2str(r) ' =c']);
     catemp = ca;
     ctemp = c;
     %eval(['D' num2str(r) ' =d']);
epsilon=1E-10;
CN=abs(catemp);
%NBabs=(CN - max(CN)) / (min(CN) - max(CN)+epsilon); % backward normalisation
NFabs=(CN - min(CN)) / (max(CN) - min(CN)+epsilon);  % Forward normalisation
for i=1:1:cb
    if NFabs(1,i)>=1
       NFabs(1,i)=1;
    end
end

% GBT Calculation
W=zeros(rb,cb);
for c1=1:1:cb-1
        W(c1,(c1+1))=1;
        W((c1+1),c1)=1;  
end
%{
WDabs=zeros(rb,cb);
for i=1:1:cb
    WDabs(i,i)=NFabs(i);
    if NFabs(i)==0
       WDabs(i,i)=epsilon;
    end
end
%}
WFabs=W;

T=zeros(rb,cb);
for j=1:1:8
    T(j,j)=ceil(sum(WFabs(j,:)));
end
%{
TDabs=zeros(rb,cb);
for j=1:1:8
    TDabs(j,j)=ceil(sum(WDabs(j,:)));
end
%}
TFabs=T;
LFabs=TFabs-WFabs;
[VFabs,DFabs]=eig(LFabs);
eval(['V1GBTA' num2str(r) ' =VFabs']);
GBTadptFA= (VFabs')*(ctemp');
GBTadptFAT=GBTadptFA';
GBTadptFAtemp=zeros(rb,cb);
GBTadptFFA(r,:) = GBTadptFAT;
GBTadptFFA=GBTadptFAtemp+GBTadptFFA;
end


if mode < 18
    GBTadptFFA=GBTadptFFA;
else
    GBTadptFFA=GBTadptFFA';
end

if mode < 18

%%%% HORIZONTAL-Veritical Calculation Start %%%%

%fprintf('Horizonal mode = %i\n', mode)

% GBT Original Block Vector Calculation Start
GBTadptFFAT=GBTadptFFA';
vGBTadptFFAT=bberes';
rberes=size(vGBTadptFFAT,1);
cberes=size(vGBTadptFFAT,2);
for r=1:1:rberes
     vda=vGBTadptFFAT(r,:);
     da=GBTadptFFAT(r,:);
     %eval(['COL' num2str(r) ' =d']);
     vdatemp = vda;
     datemp = da;
     %eval(['D' num2str(r) ' =d']);
epsilon=1E-10;
DN=abs(vdatemp);
%NBabs=(CN - max(CN)) / (min(CN) - max(CN)+epsilon); % backward normalisation
NDFabs=(DN - min(DN)) / (max(DN) - min(DN)+epsilon);  % Forward normalisation
for i=1:1:cb
    if NDFabs(1,i)>=1
       NDFabs(1,i)=1;
    end
end

W1=zeros(rb,cb);
for c1=1:1:cb-1
        W1(c1,(c1+1))=1;
        W1((c1+1),c1)=1;  
end

W1Dabs=zeros(rb,cb);
for i=1:1:cb
    W1Dabs(i,i)=NDFabs(i);
    if NDFabs(i)==0
       W1Dabs(i,i)=epsilon;
    end
end

W1Fabs=W1+W1Dabs;

T1=zeros(rb,cb);

for j=1:1:8
    T1(j,j)=ceil(sum(W1Fabs(j,:)));
end

T1Dabs=zeros(rb,cb);
for j=1:1:8
    T1Dabs(j,j)=ceil(sum(W1Dabs(j,:)));
end

T1Fabs=T1+T1Dabs;
L1Fabs=T1Fabs-W1Fabs;
[V1Fabs,D1Fabs]=eig(L1Fabs);
eval(['V2GBTA' num2str(r) ' =V1Fabs']);
GBTadptH= (V1Fabs')*(datemp');
GBTadptHT=GBTadptH';
GBTadptHH=zeros(rb,cb);
GBTadptFinaleres(r,:) = GBTadptHT;
GBTadptFinaleres=GBTadptHH+GBTadptFinaleres;
end
GBTadptFinaleres=GBTadptFinaleres';

% GBT Original Block Vector Calculation End

%{
% GBT Inermediate Block Vector Calculation Start

GBTadptFFAMT=GBTadptFFA';
rberes=size(GBTadptFFAMT,1);
cberes=size(GBTadptFFAMT,2);
for r=1:1:rberes
     dam=GBTadptFFAMT(r,:);
     %eval(['COL' num2str(r) ' =d']);
     damtemp = dam;
     %eval(['D' num2str(r) ' =d']);
epsilon=1E-10;
DN=abs(damtemp);
%NBabs=(CN - max(CN)) / (min(CN) - max(CN)+epsilon); % backward normalisation
NDFabs=(DN - min(DN)) / (max(DN) - min(DN)+epsilon);  % Forward normalisation
for i=1:1:cb
    if NDFabs(1,i)>=1
       NDFabs(1,i)=1;
    end
end

W1=zeros(rb,cb);
for c1=1:1:cb-1
        W1(c1,(c1+1))=1;
        W1((c1+1),c1)=1;  
end

W1Dabs=zeros(rb,cb);
for i=1:1:cb
    W1Dabs(i,i)=NDFabs(i);
    if NDFabs(i)==0
       W1Dabs(i,i)=epsilon;
    end
end

W1Fabs=W1+W1Dabs;

T1=zeros(rb,cb);

for j=1:1:8
    T1(j,j)=ceil(sum(W1Fabs(j,:)));
end

T1Dabs=zeros(rb,cb);
for j=1:1:8
    T1Dabs(j,j)=ceil(sum(W1Dabs(j,:)));
end

T1Fabs=T1+T1Dabs;
L1Fabs=T1Fabs-W1Fabs;
[V1mFabs,D1mFabs]=eig(L1Fabs);
eval(['V2GBTAM' num2str(r) ' =V1mFabs']);
GBTadptmH= (V1mFabs')*(damtemp');
GBTadptmHT=GBTadptmH';
GBTadptmHH=zeros(rb,cb);
GBTadptmFinaleres(r,:) = GBTadptmHT;
GBTadptmFinaleres=GBTadptmHH+GBTadptmFinaleres;
end
GBTadptmFinaleres=GBTadptmFinaleres';


% GBT Inermediate Block Vector Calculation End

%}

%%%% HORIZONTAL-Veritical Calculation End %%%%


else

%fprintf('Verical mode = %i\n', mode)

%%%% VERTICAL-Horizontal Calculation Start %%%%

% GBT Original Block Vector Calculation Start

vGBTadptFFA=bberes;
rberes=size(vGBTadptFFA,1);
cberes=size(vGBTadptFFA,2);
for r=1:1:rberes
     vda=vGBTadptFFA(r,:);
     da=GBTadptFFA(r,:);
     %eval(['COL' num2str(r) ' =d']);
     vdatemp = vda;
     datemp = da;
     %eval(['D' num2str(r) ' =d']);
epsilon=1E-10;

DN=abs(vdatemp);
%NBabs=(CN - max(CN)) / (min(CN) - max(CN)+epsilon); % backward normalisation
NDFabs=(DN - min(DN)) / (max(DN) - min(DN)+epsilon);  % Forward normalisation
for i=1:1:cb
    if NDFabs(1,i)>=1
       NDFabs(1,i)=1;
    end
end

W1=zeros(rb,cb);
for c1=1:1:cb-1
        W1(c1,(c1+1))=1;
        W1((c1+1),c1)=1;  
end

W1Dabs=zeros(rb,cb);
for i=1:1:cb
    W1Dabs(i,i)=NDFabs(i);
    if NDFabs(i)==0
       W1Dabs(i,i)=epsilon;
    end
end

W1Fabs=W1+W1Dabs;

T1=zeros(rb,cb);
for j=1:1:8
    T1(j,j)=ceil(sum(W1Fabs(j,:)));
end

T1Dabs=zeros(rb,cb);
for j=1:1:8
    T1Dabs(j,j)=ceil(sum(W1Dabs(j,:)));
end

T1Fabs=T1+T1Dabs;
L1Fabs=T1Fabs-W1Fabs;
[V1Fabs,D1Fabs]=eig(L1Fabs);
eval(['V2GBTA' num2str(r) ' =V1Fabs']);
GBTadptH= (V1Fabs')*(datemp');
GBTadptHT=GBTadptH';
GBTadptHH=zeros(rb,cb);
GBTadptFinaleres(r,:) = GBTadptHT;
GBTadptFinaleres=GBTadptHH+GBTadptFinaleres;
end


% GBT Original Block Vector Calculation End

%{
% GBT Inermediate Block Vector Calculation Start

rberes=size(GBTadptFFA,1);
cberes=size(GBTadptFFA,2);
for r=1:1:rberes
     dam=GBTadptFFA(r,:);
     %eval(['COL' num2str(r) ' =d']);
     damtemp = dam;
     %eval(['D' num2str(r) ' =d']);
epsilon=1E-10;

DN=abs(damtemp);
%NBabs=(CN - max(CN)) / (min(CN) - max(CN)+epsilon); % backward normalisation
NDFabs=(DN - min(DN)) / (max(DN) - min(DN)+epsilon);  % Forward normalisation
for i=1:1:cb
    if NDFabs(1,i)>=1
       NDFabs(1,i)=1;
    end
end

W1=zeros(rb,cb);
for c1=1:1:cb-1
        W1(c1,(c1+1))=1;
        W1((c1+1),c1)=1;  
end

W1Dabs=zeros(rb,cb);
for i=1:1:cb
    W1Dabs(i,i)=NDFabs(i);
    if NDFabs(i)==0
       W1Dabs(i,i)=epsilon;
    end
end

W1Fabs=W1+W1Dabs;

T1=zeros(rb,cb);
for j=1:1:8
    T1(j,j)=ceil(sum(W1Fabs(j,:)));
end

T1Dabs=zeros(rb,cb);
for j=1:1:8
    T1Dabs(j,j)=ceil(sum(W1Dabs(j,:)));
end

T1Fabs=T1+T1Dabs;
L1Fabs=T1Fabs-W1Fabs;
[V1mFabs,D1mFabs]=eig(L1Fabs);
eval(['V2GBTAM' num2str(r) ' =V1mFabs']);
GBTadptmH= (V1mFabs')*(damtemp');
GBTadptmHT=GBTadptmH';
GBTadptmHH=zeros(rb,cb);
GBTadptmFinaleres(r,:) = GBTadptmHT;
GBTadptmFinaleres=GBTadptmHH+GBTadptmFinaleres;
end

% GBT Inermediate Block Vector Calculation End
%}
%%%% VERTICAL-Horizontal Calculation End %%%%

end
        
       
        
     % NN GRAPH-BASED USING ACTUAL RESIDUAL
        
        ytrNN=(NNV')*x;
        
        yrNN=reshape(ytrNN,rb,cb); %These are the coeffcients obtained by using graph-based transform
        
         if abs(sum(sum(KLTres.^2)) - sum(sum(yrNN.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (actual) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             yreNN=sum(sum(yrNN.^2));
         end     
        
         % NN GRAPH-BASED USING predicted RESIDUAL
        
        ytrNN_pred=(NNV_pred')*x;
        
        yrNN_pred=reshape(ytrNN_pred,rb,cb); %These are the coeffcients obtained by using graph-based transform
        
         if abs(sum(sum(KLTres.^2)) - sum(sum(yrNN_pred.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (actual) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             yreNN_pred=sum(sum(yrNN_pred.^2));
         end     
         
        
     
        
        %GRAPH-BASED USING ACTUAL RESIDUAL
        
        ytr=(Vr')*x;
        
        yr=reshape(ytr,rb,cb); %These are the coeffcients obtained by using graph-based transform
        
         if abs(sum(sum(KLTres.^2)) - sum(sum(yr.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (actual) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             yre=sum(sum(yr.^2));
         end
    %{     
            %GRAPH-BASED USING ACTUAL RESIDUAL with 8 connectivity
            
           ytrac=(Vac')*x;
        
        yrac=reshape(ytrac,rb,cb); %These are the coeffcients obtained by using graph-based transform with 8 connectivity
        
         if abs(sum(sum(KLTres.^2)) - sum(sum(yrac.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (actual) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             yreac=sum(sum(yrac.^2));
         end   
            
    %}        
            
        
     %GRAPH-BASED USING ACTUAL RESIDUAL 2D self loop
        
        ytr_2D=(Vr_2D')*x;
        
        yr_2D=reshape(ytr_2D,rb,cb); %These are the coeffcients obtained by using graph-based transform
        
         if abs(sum(sum(KLTres.^2)) - sum(sum(yr_2D.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (actual) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             yre_2D=sum(sum(yr_2D.^2));
         end     
         
     ytr_2D=(Vr_2D')*x;
        
        yr_2D=reshape(ytr_2D,rb,cb); %These are the coeffcients obtained by using graph-based transform
        
         if abs(sum(sum(KLTres.^2)) - sum(sum(yr_2D.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (actual) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             yre_2D=sum(sum(yr_2D.^2));
         end     
              
     
        ytr_2D_old=(Vr_2D_old')*x;
        
        yr_2D_old=reshape(ytr_2D_old,rb,cb); %These are the coeffcients obtained by using graph-based transform
        
         if abs(sum(sum(KLTres.^2)) - sum(sum(yr_2D_old.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (actual) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             yre_2D=sum(sum(yr_2D_old.^2));
         end     
         
     ytr_2D_old=(Vr_2D_old')*x;
        
        yr_2D_old=reshape(ytr_2D_old,rb,cb); %These are the coeffcients obtained by using graph-based transform
        
         if abs(sum(sum(KLTres.^2)) - sum(sum(yr_2D_old.^2))) > 1
             rBlock;
             cBlock;
             disp('Warning - energy of GBT (actual) greater than that of KLT');
             KLTe=sum(sum(KLTres.^2));
             yre_2D=sum(sum(yr_2D_old.^2));
         end     
                 
         
         
    %GRAPH-BASED USING ACTUAL RESIDUAL Separable
    
    
if mode < 18
    bb=resBlock;  %% Horizontal Block
    fprintf('Horizonal mode = %i\n', mode)
else
    bb=resBlock'; %% Vertical Block
    fprintf('Vertical mode = %i\n', mode)
end
         
         
rb=size(bb,1);
cb=size(bb,2);

for r=1:1:rb
     c=bb(r,:);
   % eval(['ROW' num2str(r) ' =c']);
     ctemp = c;
   % eval(['D' num2str(r) ' =d']);
epsilon=1E-10;
CN=abs(ctemp);
%NBabs=(CN - max(CN)) / (min(CN) - max(CN)+epsilon); % backward normalisation
NFabs=(CN - min(CN)) / (max(CN) - min(CN)+epsilon);  % Forward normalisation
for i=1:1:cb
    if NFabs(1,i)>=1
       NFabs(1,i)=1;
    end
end


W=zeros(rb,cb);
for c1=1:1:cb-1
        W(c1,(c1+1))=1;
        W((c1+1),c1)=1;  
end
%{
WDabs=zeros(rb,cb);
for i=1:1:cb
    WDabs(i,i)=NFabs(i);
    if NFabs(i)==0
       WDabs(i,i)=epsilon;
    end
end
%}
WFabs=W;

T=zeros(rb,cb);
TW=zeros(rb,cb);
for j=1:1:8
    T(j,j)=ceil(sum(WFabs(j,:)));
end
%{
TDabs=zeros(rb,cb);
for j=1:1:8
    TDabs(j,j)=ceil(sum(WDabs(j,:)));
end
%}
TFabs=T;
LFabs=TFabs-WFabs;
[VFabs,DFabs]=eig(LFabs);
eval(['V1GBT' num2str(r) ' =VFabs']);
GBTadptF= (VFabs')*(ctemp');
GBTadptFT=GBTadptF';
GBTadptFtemp=zeros(rb,cb);
GBTadptFF(r,:) = GBTadptFT;
GBTadptFF=GBTadptFtemp+GBTadptFF;
end



if mode < 18
%    KLTresF=KLTresF;
%    DCTresF=DCTresF;
%    DSTresF=DSTresF;
    GBTadptFF=GBTadptFF;
else
%    KLTresF=KLTresF';
%    DCTresF=DCTresF';
%    DSTresF=DSTresF';
    GBTadptFF=GBTadptFF';
end

% End calculation of initial input block %
         
         
         
if mode < 18

%%%% HORIZONTAL-Veritical Calculation Start %%%%

%fprintf('Horizonal mode = %i\n', mode)

% GBT Calculation
GBTadptFFT=GBTadptFF';
rb=size(GBTadptFFT,1);
cb=size(GBTadptFFT,2);
for r=1:1:rb
     d=GBTadptFFT(r,:);
     %eval(['COL' num2str(r) ' =d']);
     dtemp = d;
     %eval(['D' num2str(r) ' =d']);
epsilon=1E-10;

DN=abs(dtemp);
%NBabs=(CN - max(CN)) / (min(CN) - max(CN)+epsilon); % backward normalisation
NDFabs=(DN - min(DN)) / (max(DN) - min(DN)+epsilon);  % Forward normalisation
for i=1:1:cb
    if NDFabs(1,i)>=1
       NDFabs(1,i)=1;
    end
end

W1=zeros(rb,cb);
for c1=1:1:cb-1
        W1(c1,(c1+1))=1;
        W1((c1+1),c1)=1;  
end

W1Dabs=zeros(rb,cb);
for i=1:1:cb
    W1Dabs(i,i)=NDFabs(i);
    if NDFabs(i)==0
       W1Dabs(i,i)=epsilon;
    end
end

W1Fabs=W1+W1Dabs;

T1=zeros(rb,cb);

for j=1:1:8
    T1(j,j)=ceil(sum(W1Fabs(j,:)));
end

T1Dabs=zeros(rb,cb);
for j=1:1:8
    T1Dabs(j,j)=ceil(sum(W1Dabs(j,:)));
end

T1Fabs=T1+T1Dabs;
L1Fabs=T1Fabs-W1Fabs;
[V1Fabs,D1Fabs]=eig(L1Fabs);
eval(['V2GBT' num2str(r) ' =V1Fabs']);
GBTadptH= (V1Fabs')*(dtemp');
GBTadptHT=GBTadptH';
GBTadptHH=zeros(rb,cb);
GBTadptFinal(r,:) = GBTadptHT;
GBTadptFinal=GBTadptHH+GBTadptFinal;
end
GBTadptFinal=GBTadptFinal';


else

%fprintf('Verical mode = %i\n', mode)

%%%% VERTICAL-Horizontal Calculation Start %%%%

% GBT Calculation
rb=size(GBTadptFF,1);
cb=size(GBTadptFF,2);
for r=1:1:rb
     d=GBTadptFF(r,:);
     %eval(['COL' num2str(r) ' =d']);
     dtemp = d;
     %eval(['D' num2str(r) ' =d']);
epsilon=1E-10;
DN=abs(dtemp);
%NBabs=(CN - max(CN)) / (min(CN) - max(CN)+epsilon); % backward normalisation
NDFabs=(DN - min(DN)) / (max(DN) - min(DN)+epsilon);  % Forward normalisation
for i=1:1:cb
    if NDFabs(1,i)>=1
       NDFabs(1,i)=1;
    end
end

W1=zeros(rb,cb);
for c1=1:1:cb-1
        W1(c1,(c1+1))=1;
        W1((c1+1),c1)=1;  
end

W1Dabs=zeros(rb,cb);
for i=1:1:cb
    W1Dabs(i,i)=NDFabs(i);
    if NDFabs(i)==0
       W1Dabs(i,i)=epsilon;
    end
end

W1Fabs=W1+W1Dabs;

T1=zeros(rb,cb);
for j=1:1:8
    T1(j,j)=ceil(sum(W1Fabs(j,:)));
end

T1Dabs=zeros(rb,cb);
for j=1:1:8
    T1Dabs(j,j)=ceil(sum(W1Dabs(j,:)));
end

T1Fabs=T1+T1Dabs;
L1Fabs=T1Fabs-W1Fabs;
[V1Fabs,D1Fabs]=eig(L1Fabs);
eval(['V2GBT' num2str(r) ' =V1Fabs']);
GBTadptH= (V1Fabs')*(dtemp');
GBTadptHT=GBTadptH';
GBTadptHH=zeros(rb,cb);
GBTadptFinal(r,:) = GBTadptHT;
GBTadptFinal=GBTadptHH+GBTadptFinal;
end

%%%% VERTICAL-Horizontal Calculation End %%%%


end

         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
        %SELECT LARGEST-VALUED COEFFCIENTS FOR EACH TRANSFORM BASED ON AN
        %ENERGY BUDGET
        
        maxMSE=sum(sum((resBlock).^2));
        %eTotal=5*sum(sum(KLTres.^2));
        eTotal=sum(sum(KLTres.^2)); %THIS IS MY ENERGY BUDGET. THE ENERGY OF THE KLT COEFFICIENTS.
        step=0.25;
        %step=1;
        %%DCT coefficients
        im=DCTres;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEDCT(nCoeff+1) = eTmpBlock/eTotal;   
            
            resLT=idct2(tmpBlock);
            
            mses(rBlock,cBlock).MSEDCT(nCoeff+1) = sum(sum((resBlock-resLT).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEDCT(nCoeff+1)>1
                energies(rBlock,cBlock).PEDCT(nCoeff+1)=1;
      %          mses(rBlock,cBlock).MSEDCT(nCoeff+1)=NaN;%MSEDCT(nCoeff);
            end
            
        end
        
        energies(rBlock,cBlock).PEDCT=smooth_PE(energies(rBlock,cBlock).PEDCT);
        mses(rBlock,cBlock).MSEDCT=smooth_MSE(mses(rBlock,cBlock).MSEDCT);
        
        %KLT coefficients
        
        im=KLTres;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEKLT(nCoeff+1) = eTmpBlock/eTotal;
%             
%             if energies(rBlock,cBlock).PEKLT(nCoeff+1)==0
%              th
%             end
            
%             for l=1:8
%                 inv_trans_img(:,l)=vv{m-1}'*tmpBlock(:,l);
%             end
            
%             resLT4=inv_trans_img;

            tmpBlock = reshape(tmpBlock,rb*cb,1);
            resLT4=(vvv)*tmpBlock;
            resLT4 = reshape(resLT4,rb,cb);

            mses(rBlock,cBlock).MSEKLT(nCoeff+1) = sum(sum((resBlock-resLT4).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEKLT(nCoeff+1)>1
                energies(rBlock,cBlock).PEKLT(nCoeff+1)=1;
             %   mses(rBlock,cBlock).MSEKLT(nCoeff+1)=NaN;%MSEKLT(nCoeff);
            end
            
        end
        
        energies(rBlock,cBlock).PEKLT=smooth_PE(energies(rBlock,cBlock).PEKLT);
        mses(rBlock,cBlock).MSEKLT=smooth_MSE(mses(rBlock,cBlock).MSEKLT);
        
        % GBT-ONL coefficients
        
          im=ONLV_R_R;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEGBTONLLap(nCoeff+1) = eTmpBlock/eTotal;
%             
%             if energies(rBlock,cBlock).PEKLT(nCoeff+1)==0
%              th
%             end
            
%             for l=1:8
%                 inv_trans_img(:,l)=vv{m-1}'*tmpBlock(:,l);
%             end
            
%             resLT4=inv_trans_img;

            tmpBlock = reshape(tmpBlock,rb*cb,1);
            GBTONLLap2=(ONLV)*tmpBlock;
            GBTONLLap2 = reshape(GBTONLLap2,rb,cb);

            mses(rBlock,cBlock).MSEGBTONLLap(nCoeff+1) = sum(sum((resBlock-GBTONLLap2).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEGBTONLLap(nCoeff+1)>1
                energies(rBlock,cBlock).PEGBTONLLap(nCoeff+1)=1;
             %   mses(rBlock,cBlock).MSEKLT(nCoeff+1)=NaN;%MSEKLT(nCoeff);
            end
            
        end
        
        energies(rBlock,cBlock).PEGBTONLLap=smooth_PE(energies(rBlock,cBlock).PEGBTONLLap);
        mses(rBlock,cBlock).MSEGBTONLLap=smooth_MSE(mses(rBlock,cBlock).MSEGBTONLLap);
        
        
        
        
        
        
        
        
        
        
        
        
        
        %Graph-based - aproximation with Template Matching type A
        
        im=y;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEG(nCoeff+1) = eTmpBlock/eTotal;
            
            
            
            yt2 = reshape(tmpBlock,rb*cb,1);
            resLTG=(V)*yt2;
            
            resLTG = reshape(resLTG,rb,cb);
           % resLTG=round(resLTG);
            
            mses(rBlock,cBlock).MSEG(nCoeff+1)= sum(sum((resBlock-resLTG).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEG(nCoeff+1)>1
                energies(rBlock,cBlock).PEG(nCoeff+1)=1;
           %     mses(rBlock,cBlock).MSEG(nCoeff+1)=NaN;%MSEG(nCoeff);
            end
    
        end
        
        energies(rBlock,cBlock).PEG=smooth_PE(energies(rBlock,cBlock).PEG);
        mses(rBlock,cBlock).MSEG=smooth_MSE(mses(rBlock,cBlock).MSEG);
    
     %Graph-based - aproximation with Ortega's method
        
        im=y_ORT;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEG_ORT(nCoeff+1) = eTmpBlock/eTotal;
            
            
            
            yt2_ORT = reshape(tmpBlock,rb*cb,1);
            resLTG_ORT=(ORTV)*yt2_ORT;
            
            resLTG_ORT = reshape(resLTG_ORT,rb,cb);
           % resLTG=round(resLTG);
            
            mses(rBlock,cBlock).MSEG_ORT(nCoeff+1)= sum(sum((resBlock-resLTG_ORT).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEG_ORT(nCoeff+1)>1
                energies(rBlock,cBlock).PEG_ORT(nCoeff+1)=1;
           %     mses(rBlock,cBlock).MSEG(nCoeff+1)=NaN;%MSEG(nCoeff);
            end
    
        end
        
        energies(rBlock,cBlock).PEG_ORT=smooth_PE(energies(rBlock,cBlock).PEG_ORT);
        mses(rBlock,cBlock).MSEG_ORT=smooth_MSE(mses(rBlock,cBlock).MSEG_ORT);
            
          
        
        
    %Graph-based - aproximation with Template Matching type A with 2D self loop
    
        
        im=y_2D;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEG_2D(nCoeff+1) = eTmpBlock/eTotal;
            
            
            
            yt2_2D = reshape(tmpBlock,rb*cb,1);
            resLTG_2D=(V_2D)*yt2_2D;
            
            resLTG_2D = reshape(resLTG_2D,rb,cb);
           % resLTG=round(resLTG);
            
            mses(rBlock,cBlock).MSEG_2D(nCoeff+1)= sum(sum((resBlock-resLTG_2D).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEG_2D(nCoeff+1)>1
                energies(rBlock,cBlock).PEG_2D(nCoeff+1)=1;
           %     mses(rBlock,cBlock).MSEG(nCoeff+1)=NaN;%MSEG(nCoeff);
            end
    
        end
        
        energies(rBlock,cBlock).PEG_2D=smooth_PE(energies(rBlock,cBlock).PEG_2D);
        mses(rBlock,cBlock).MSEG_2D=smooth_MSE(mses(rBlock,cBlock).MSEG_2D);
             
        
         %Graph-based - aproximation with Template Matching type A NLM
        
        im=ynlm;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEG_NLM(nCoeff+1) = eTmpBlock/eTotal;
            
            
            
            yt2nlm = reshape(tmpBlock,rb*cb,1);
            resLTGnlm=(V_TMANLM)*yt2nlm;
            
            resLTGnlm = reshape(resLTGnlm,rb,cb);
           % resLTG=round(resLTG);
            
            mses(rBlock,cBlock).MSEG_NLM(nCoeff+1)= sum(sum((resBlock-resLTGnlm).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEG_NLM(nCoeff+1)>1
                energies(rBlock,cBlock).PEG_NLM(nCoeff+1)=1;
           %     mses(rBlock,cBlock).MSEG(nCoeff+1)=NaN;%MSEG(nCoeff);
            end
    
        end
        
        energies(rBlock,cBlock).PEG_NLM=smooth_PE(energies(rBlock,cBlock).PEG_NLM);
        mses(rBlock,cBlock).MSEG_NLM=smooth_MSE(mses(rBlock,cBlock).MSEG_NLM);
        
        
   %Graph-based - aproximation with Template Matching type A NLM with 2D GBT self loop
 
        
        im=ynlm_2D;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEG_NLM_2D(nCoeff+1) = eTmpBlock/eTotal;
            
            
            
            yt2nlm_2D = reshape(tmpBlock,rb*cb,1);
            resLTGnlm_2D=(V_TMANLM_2D)*yt2nlm_2D;
            
            resLTGnlm_2D = reshape(resLTGnlm_2D,rb,cb);
           % resLTG=round(resLTG);
            
            mses(rBlock,cBlock).MSEG_NLM_2D(nCoeff+1)= sum(sum((resBlock-resLTGnlm_2D).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEG_NLM_2D(nCoeff+1)>1
                energies(rBlock,cBlock).PEG_NLM_2D(nCoeff+1)=1;
           %     mses(rBlock,cBlock).MSEG(nCoeff+1)=NaN;%MSEG(nCoeff);
            end
    
        end
        
        energies(rBlock,cBlock).PEG_NLM_2D=smooth_PE(energies(rBlock,cBlock).PEG_NLM_2D);
        mses(rBlock,cBlock).MSEG_NLM_2D=smooth_MSE(mses(rBlock,cBlock).MSEG_NLM_2D);
               
        
        
        
        
        
        
        
        %Graph-based - aproximation with Template Matching type A separable 
        
        im=TMFinaleres;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            TMtmpBlock=mask.*im;
            TMeTmpBlock=sum(sum(TMtmpBlock.^2));
            eTmpBlockGA=TMeTmpBlock;
            nCoeff=sum(sum(mask));
            
      %      energies(rBlock,cBlock).PEGTMFA(nCoeff+1) = eTmpBlock/eTotaleres;
      energies(rBlock,cBlock).PEGTMFA(nCoeff+1) = TMeTmpBlock/eTotal;
      
 

      
dd1=TMtmpBlock;
if mode < 18
dd1=dd1';
rb=size(dd1,1);
cb=size(dd1,2);
for r=1:1:rb
     d1=dd1(r,:);
%     %eval(['DCTI' num2str(r) ' =d1']);
     TMempI = d1;
     %  d=dct2(c);% DCT Calculation
if r==1
    G2IDA=V2TMA1;
end
if r==2
    G2IDA=V2TMA2;
end
if r==3
    G2IDA=V2TMA3;
end
if r==4
    G2IDA=V2TMA4;
end
if r==5
    G2IDA=V2TMA5;
end
if r==6
    G2IDA=V2TMA6;
end
if r==7
    G2IDA=V2TMA7;
end
if r==8
    G2IDA=V2TMA8;
end 

resTMa1=(G2IDA)*TMempI';           
TMresFHIA = resTMa1';     
TMresFH1IA=zeros(rb,cb);
TMresFIA(r,:) = TMresFHIA;
TMresFIA=TMresFH1IA+TMresFIA;
end
TMresFIA=TMresFIA';
  
bb1=TMresFIA;
rb=size(bb1,1);
cb=size(bb1,2);
for r=1:1:rb
     c1=bb1(r,:);
     ctempI = c1;
if r==1
    G1IDA=V1TMA1;
end
if r==2
    G1IDA=V1TMA2;
end
if r==3
    G1IDA=V1TMA3;
end
if r==4
    G1IDA=V1TMA4;
end
if r==5
    G1IDA=V1TMA5;
end
if r==6
    G1IDA=V1TMA6;
end
if r==7
    G1IDA=V1TMA7;
end
if r==8
    G1IDA=V1TMA8;
end
     
resTMa2=ctempI*(G1IDA)';
TMresFtempIA=resTMa2';
TMresF1IIA=zeros(rb,cb);
resBlockTMA(r,:) = TMresFtempIA;
resBlockTMA=TMresF1IIA+resBlockTMA;
end


else
rb=size(dd1,1);
cb=size(dd1,2);
for r=1:1:rb
     d1=dd1(r,:);
     TMempI = d1;
if r==1
    G2IDA=V2TMA1;
end
if r==2
    G2IDA=V2TMA2;
end
if r==3
    G2IDA=V2TMA3;
end
if r==4
    G2IDA=V2TMA4;
end
if r==5
    G2IDA=V2TMA5;
end
if r==6
    G2IDA=V2TMA6;
end
if r==7
    G2IDA=V2TMA7;
end
if r==8
    G2IDA=V2TMA8;
end 

resTMa1=(G2IDA)*TMempI';          
TMresFHIA = resTMa1';     
TMresFH1IA=zeros(rb,cb);
TMresFIA(r,:) = TMresFHIA;
TMresFIA=TMresFH1IA+TMresFIA;
end

TMresFIA=TMresFIA';
bb1=TMresFIA;
rb=size(bb1,1);
cb=size(bb1,2);
for r=1:1:rb
     c1=bb1(r,:);
     ctempI = c1;
if r==1
    G1IDA=V1TMA1;
end
if r==2
    G1IDA=V1TMA2;
end
if r==3
    G1IDA=V1TMA3;
end
if r==4
    G1IDA=V1TMA4;
end
if r==5
    G1IDA=V1TMA5;
end
if r==6
    G1IDA=V1TMA6;
end
if r==7
    G1IDA=V1TMA7;
end
if r==8
    G1IDA=V1TMA8;
end
     
resTMa2=ctempI*(G1IDA)';
TMresFtempIA=resTMa2';
TMresF1IIA=zeros(rb,cb);
resBlockTMA(r,:) = TMresFtempIA;
resBlockTMA=TMresF1IIA+resBlockTMA;
end
resBlockTMA=resBlockTMA';
end            
mses(rBlock,cBlock).MSETMFA(nCoeff+1) = sum(sum((resBlock-resBlockTMA).^2))/maxMSE; 
 
     
      
      
       
        if energies(rBlock,cBlock).PEGTMFA(nCoeff+1)>1
           energies(rBlock,cBlock).PEGTMFA(nCoeff+1)=1;
        end
        
        end
        energies(rBlock,cBlock).PEGTMFA=smooth_PE(energies(rBlock,cBlock).PEGTMFA);
        mses(rBlock,cBlock).MSETMFA=smooth_MSE(mses(rBlock,cBlock).MSETMFA);   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        %Graph-based - aproximation with Template Matching TYPE B
        
        im=y_TMB;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEG_TMB(nCoeff+1) = eTmpBlock/eTotal;
            
            
            
            yt2_TMB = reshape(tmpBlock,rb*cb,1);
            resLTG_TMB=(V_TMB)*yt2_TMB;
            
            resLTG_TMB = reshape(resLTG_TMB,rb,cb);
          %  resLTG=round(resLTG);
            
            mses(rBlock,cBlock).MSEG_TMB(nCoeff+1)= sum(sum((resBlock-resLTG_TMB).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEG_TMB(nCoeff+1)>1
                energies(rBlock,cBlock).PEG_TMB(nCoeff+1)=1;
           %     mses(rBlock,cBlock).MSEG(nCoeff+1)=NaN;%MSEG(nCoeff);
            end
    
        end
        
        energies(rBlock,cBlock).PEG_TMB=smooth_PE(energies(rBlock,cBlock).PEG_TMB);
        mses(rBlock,cBlock).MSEG_TMB=smooth_MSE(mses(rBlock,cBlock).MSEG_TMB); 
        
  %Graph-based -aproximation with Template Matching TYPE B with 2DGBT self
  %loop
        
        im=y_TMB_2D;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEG_TMB_2D(nCoeff+1) = eTmpBlock/eTotal;
            
            
            
            yt2_TMB_2D = reshape(tmpBlock,rb*cb,1);
            resLTG_TMB_2D=(V_TMB_2D)*yt2_TMB_2D;
            
            resLTG_TMB_2D = reshape(resLTG_TMB_2D,rb,cb);
          %  resLTG=round(resLTG);
            
            mses(rBlock,cBlock).MSEG_TMB_2D(nCoeff+1)= sum(sum((resBlock-resLTG_TMB_2D).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEG_TMB_2D(nCoeff+1)>1
                energies(rBlock,cBlock).PEG_TMB_2D(nCoeff+1)=1;
           %     mses(rBlock,cBlock).MSEG(nCoeff+1)=NaN;%MSEG(nCoeff);
            end
    
        end
        
        energies(rBlock,cBlock).PEG_TMB_2D=smooth_PE(energies(rBlock,cBlock).PEG_TMB_2D);
        mses(rBlock,cBlock).MSEG_TMB_2D=smooth_MSE(mses(rBlock,cBlock).MSEG_TMB_2D); 
        
         

         %Graph-based - aproximation with Template Matching type B NLM
        
        im=y_TMBnlm;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEG_TMBnlm(nCoeff+1) = eTmpBlock/eTotal;
            
            
            
            yt2_TMBnlm = reshape(tmpBlock,rb*cb,1);
            resLTG_TMBnlm=(V_TMBNLM)*yt2_TMBnlm;
            
            resLTG_TMBnlm = reshape(resLTG_TMBnlm,rb,cb);
          %  resLTG=round(resLTG);
            
            mses(rBlock,cBlock).MSEG_TMBnlm(nCoeff+1)= sum(sum((resBlock-resLTG_TMBnlm).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEG_TMBnlm(nCoeff+1)>1
                energies(rBlock,cBlock).PEG_TMBnlm(nCoeff+1)=1;
           %     mses(rBlock,cBlock).MSEG(nCoeff+1)=NaN;%MSEG(nCoeff);
            end
    
        end
        
        energies(rBlock,cBlock).PEG_TMBnlm=smooth_PE(energies(rBlock,cBlock).PEG_TMBnlm);
        mses(rBlock,cBlock).MSEG_TMBnlm=smooth_MSE(mses(rBlock,cBlock).MSEG_TMBnlm);
        
   %Graph-based - aproximation with Template Matching type B NLM 2D self loop
  
        
        im=y_TMBnlm_2D;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEG_TMBnlm_2D(nCoeff+1) = eTmpBlock/eTotal;
            
            
            
            yt2_TMBnlm_2D = reshape(tmpBlock,rb*cb,1);
            resLTG_TMBnlm_2D=(V_TMBNLM_2D)*yt2_TMBnlm_2D;
            
            resLTG_TMBnlm_2D = reshape(resLTG_TMBnlm_2D,rb,cb);
          %  resLTG=round(resLTG);
            
            mses(rBlock,cBlock).MSEG_TMBnlm_2D(nCoeff+1)= sum(sum((resBlock-resLTG_TMBnlm_2D).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEG_TMBnlm_2D(nCoeff+1)>1
                energies(rBlock,cBlock).PEG_TMBnlm_2D(nCoeff+1)=1;
           %     mses(rBlock,cBlock).MSEG(nCoeff+1)=NaN;%MSEG(nCoeff);
            end
    
        end
        
        energies(rBlock,cBlock).PEG_TMBnlm_2D=smooth_PE(energies(rBlock,cBlock).PEG_TMBnlm_2D);
        mses(rBlock,cBlock).MSEG_TMBnlm_2D=smooth_MSE(mses(rBlock,cBlock).MSEG_TMBnlm_2D);
        
         
        
          
        im=y_TMBnlm_2D_old;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEG_TMBnlm_2D_old(nCoeff+1) = eTmpBlock/eTotal;
            
            
            
            yt2_TMBnlm_2D_old = reshape(tmpBlock,rb*cb,1);
            resLTG_TMBnlm_2D_old=(V_TMBNLM_2D_old)*yt2_TMBnlm_2D_old;
            
            resLTG_TMBnlm_2D_old = reshape(resLTG_TMBnlm_2D_old,rb,cb);
          %  resLTG=round(resLTG);
            
            mses(rBlock,cBlock).MSEG_TMBnlm_2D_old(nCoeff+1)= sum(sum((resBlock-resLTG_TMBnlm_2D_old).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEG_TMBnlm_2D_old(nCoeff+1)>1
                energies(rBlock,cBlock).PEG_TMBnlm_2D_old(nCoeff+1)=1;
           %     mses(rBlock,cBlock).MSEG(nCoeff+1)=NaN;%MSEG(nCoeff);
            end
    
        end
        
        energies(rBlock,cBlock).PEG_TMBnlm_2D_old=smooth_PE(energies(rBlock,cBlock).PEG_TMBnlm_2D_old);
        mses(rBlock,cBlock).MSEG_TMBnlm_2D_old=smooth_MSE(mses(rBlock,cBlock).MSEG_TMBnlm_2D_old);
        
        
        
           %Graph-based - aproximation with Template Matching type B separable 
        
        im=TMBFinaleres;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            TMBtmpBlock=mask.*im;
            TMBeTmpBlock=sum(sum(TMBtmpBlock.^2));
            eTmpBlockGA=TMBeTmpBlock;
            nCoeff=sum(sum(mask));
            
      %      energies(rBlock,cBlock).PEGTMFA(nCoeff+1) = eTmpBlock/eTotaleres;
      energies(rBlock,cBlock).PEGTMFB(nCoeff+1) = TMBeTmpBlock/eTotal;
      
 

      
dd1=TMBtmpBlock;
if mode < 18
dd1=dd1';
rb=size(dd1,1);
cb=size(dd1,2);
for r=1:1:rb
     d1=dd1(r,:);
%     %eval(['DCTI' num2str(r) ' =d1']);
     TMempI = d1;
     %  d=dct2(c);% DCT Calculation
if r==1
    G2IDA=V2TMB1;
end
if r==2
    G2IDA=V2TMB2;
end
if r==3
    G2IDA=V2TMB3;
end
if r==4
    G2IDA=V2TMB4;
end
if r==5
    G2IDA=V2TMB5;
end
if r==6
    G2IDA=V2TMB6;
end
if r==7
    G2IDA=V2TMB7;
end
if r==8
    G2IDA=V2TMB8;
end 

resTMB1=(G2IDA)*TMempI';           
TMBresFHIA = resTMB1';     
TMBresFH1IA=zeros(rb,cb);
TMBresFIA(r,:) = TMBresFHIA;
TMBresFIA=TMBresFH1IA+TMBresFIA;
end
TMBresFIA=TMBresFIA';
  
bb1=TMBresFIA;
rb=size(bb1,1);
cb=size(bb1,2);
for r=1:1:rb
     c1=bb1(r,:);
     ctempI = c1;
if r==1
    G1IDA=V1TMB1;
end
if r==2
    G1IDA=V1TMB2;
end
if r==3
    G1IDA=V1TMB3;
end
if r==4
    G1IDA=V1TMB4;
end
if r==5
    G1IDA=V1TMB5;
end
if r==6
    G1IDA=V1TMB6;
end
if r==7
    G1IDA=V1TMB7;
end
if r==8
    G1IDA=V1TMB8;
end
     
resTMB2=ctempI*(G1IDA)';
TMBresFtempIA=resTMB2';
TMBresF1IIA=zeros(rb,cb);
resBlockTMB(r,:) = TMBresFtempIA;
resBlockTMB=TMBresF1IIA+resBlockTMB;
end


else
rb=size(dd1,1);
cb=size(dd1,2);
for r=1:1:rb
     d1=dd1(r,:);
     TMempI = d1;
if r==1
    G2IDA=V2TMB1;
end
if r==2
    G2IDA=V2TMB2;
end
if r==3
    G2IDA=V2TMB3;
end
if r==4
    G2IDA=V2TMB4;
end
if r==5
    G2IDA=V2TMB5;
end
if r==6
    G2IDA=V2TMB6;
end
if r==7
    G2IDA=V2TMB7;
end
if r==8
    G2IDA=V2TMB8;
end 

resTMB1=(G2IDA)*TMempI';          
TMBresFHIA = resTMB1';     
TMBresFH1IA=zeros(rb,cb);
TMBresFIA(r,:) = TMBresFHIA;
TMBresFIA=TMBresFH1IA+TMBresFIA;
end

TMBresFIA=TMBresFIA';
bb1=TMBresFIA;
rb=size(bb1,1);
cb=size(bb1,2);
for r=1:1:rb
     c1=bb1(r,:);
     ctempI = c1;
if r==1
    G1IDA=V1TMB1;
end
if r==2
    G1IDA=V1TMB2;
end
if r==3
    G1IDA=V1TMB3;
end
if r==4
    G1IDA=V1TMB4;
end
if r==5
    G1IDA=V1TMB5;
end
if r==6
    G1IDA=V1TMB6;
end
if r==7
    G1IDA=V1TMB7;
end
if r==8
    G1IDA=V1TMB8;
end
     
resTMB2=ctempI*(G1IDA)';
TMBresFtempIA=resTMB2';
TMBresF1IIA=zeros(rb,cb);
resBlockTMB(r,:) = TMBresFtempIA;
resBlockTMB=TMBresF1IIA+resBlockTMB;
end
resBlockTMB=resBlockTMB';
end            
mses(rBlock,cBlock).MSETMFB(nCoeff+1) = sum(sum((resBlock-resBlockTMB).^2))/maxMSE; 
 
     
      
      
       
        if energies(rBlock,cBlock).PEGTMFB(nCoeff+1)>1
           energies(rBlock,cBlock).PEGTMFB(nCoeff+1)=1;
        end
        
        end
        energies(rBlock,cBlock).PEGTMFB=smooth_PE(energies(rBlock,cBlock).PEGTMFB);
        mses(rBlock,cBlock).MSETMFB=smooth_MSE(mses(rBlock,cBlock).MSETMFB);     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
     %{   
                %Graph-based - aproximation with 8 connectivity
        
        im=ye;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEGE(nCoeff+1) = eTmpBlock/eTotal;
            
            
            
            yt2e = reshape(tmpBlock,rb*cb,1);
            resLTGE=(Ves)*yt2e;
            
            resLTGE = reshape(resLTGE,rb,cb);
            resLTGE=round(resLTGE);
            
            mses(rBlock,cBlock).MSEGE(nCoeff+1)= sum(sum((resBlock-resLTGE).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEGE(nCoeff+1)>1
                energies(rBlock,cBlock).PEGE(nCoeff+1)=NaN;
                mses(rBlock,cBlock).MSEGE(nCoeff+1)=NaN;%MSEGE(nCoeff);
            end
    
        end
        
     %}   
        
     %Graph-based - aproximation with Prediction Inaccuracy
        
        im=y_PI;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEG_PI(nCoeff+1) = eTmpBlock/eTotal;
            
            
            
            yt2_PI = reshape(tmpBlock,rb*cb,1);
            resLTG_PI=(V_PI)*yt2_PI;
            
            resLTG_PI = reshape(resLTG_PI,rb,cb);
          %  resLTG_PI=round(resLTG_PI);
            
            mses(rBlock,cBlock).MSEG_PI(nCoeff+1)= sum(sum((resBlock-resLTG_PI).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEG_PI(nCoeff+1)>1
                energies(rBlock,cBlock).PEG_PI(nCoeff+1)=1;
           %     mses(rBlock,cBlock).MSEG(nCoeff+1)=NaN;%MSEG(nCoeff);
            end
    
        end
        
        energies(rBlock,cBlock).PEG_PI=smooth_PE(energies(rBlock,cBlock).PEG_PI);
        mses(rBlock,cBlock).MSEG_PI=smooth_MSE(mses(rBlock,cBlock).MSEG_PI);
        
   %Graph-based - aproximation with Prediction Inaccuracy 2D GBT self loop
        
        im=y_PI_2D;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEG_PI_2D(nCoeff+1) = eTmpBlock/eTotal;
            
            
            
            yt2_PI_2D = reshape(tmpBlock,rb*cb,1);
            resLTG_PI_2D=(V_PI_2D)*yt2_PI_2D;
            
            resLTG_PI_2D = reshape(resLTG_PI_2D,rb,cb);
          %  resLTG_PI=round(resLTG_PI);
            
            mses(rBlock,cBlock).MSEG_PI_2D(nCoeff+1)= sum(sum((resBlock-resLTG_PI_2D).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEG_PI_2D(nCoeff+1)>1
                energies(rBlock,cBlock).PEG_PI_2D(nCoeff+1)=1;
           %     mses(rBlock,cBlock).MSEG(nCoeff+1)=NaN;%MSEG(nCoeff);
            end
    
        end
        
        energies(rBlock,cBlock).PEG_PI_2D=smooth_PE(energies(rBlock,cBlock).PEG_PI_2D);
        mses(rBlock,cBlock).MSEG_PI_2D=smooth_MSE(mses(rBlock,cBlock).MSEG_PI_2D);
        
          
 
        
        
        
        
        
        
        
        %Graph-based - aproximation separable 
        
        im=GBTadptFinaleres;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            gbtatmpBlock=mask.*im;
            gbtaeTmpBlock=sum(sum(gbtatmpBlock.^2));
            eTmpBlockGA=gbtaeTmpBlock;
            nCoeff=sum(sum(mask));
            
      %      energies(rBlock,cBlock).PEGGBTadptFA(nCoeff+1) = eTmpBlock/eTotaleres;
      energies(rBlock,cBlock).PEGGBTadptFA(nCoeff+1) = gbtaeTmpBlock/eTotal;
      
 

      
dd1=gbtatmpBlock;
if mode < 18
dd1=dd1';
rb=size(dd1,1);
cb=size(dd1,2);
for r=1:1:rb
     d1=dd1(r,:);
%     %eval(['DCTI' num2str(r) ' =d1']);
     gbtempI = d1;
     %  d=dct2(c);% DCT Calculation
if r==1
    G2IDA=V2GBTA1;
end
if r==2
    G2IDA=V2GBTA2;
end
if r==3
    G2IDA=V2GBTA3;
end
if r==4
    G2IDA=V2GBTA4;
end
if r==5
    G2IDA=V2GBTA5;
end
if r==6
    G2IDA=V2GBTA6;
end
if r==7
    G2IDA=V2GBTA7;
end
if r==8
    G2IDA=V2GBTA8;
end 

resgbta1=(G2IDA)*gbtempI';           
GBTresFHIA = resgbta1';     
GBTresFH1IA=zeros(rb,cb);
GBTresFIA(r,:) = GBTresFHIA;
GBTresFIA=GBTresFH1IA+GBTresFIA;
end
GBTresFIA=GBTresFIA';
  
bb1=GBTresFIA;
rb=size(bb1,1);
cb=size(bb1,2);
for r=1:1:rb
     c1=bb1(r,:);
     ctempI = c1;
if r==1
    G1IDA=V1GBTA1;
end
if r==2
    G1IDA=V1GBTA2;
end
if r==3
    G1IDA=V1GBTA3;
end
if r==4
    G1IDA=V1GBTA4;
end
if r==5
    G1IDA=V1GBTA5;
end
if r==6
    G1IDA=V1GBTA6;
end
if r==7
    G1IDA=V1GBTA7;
end
if r==8
    G1IDA=V1GBTA8;
end
     
resgbta2=ctempI*(G1IDA)';
GBTresFtempIA=resgbta2';
GBTresF1IIA=zeros(rb,cb);
resBlockGBTA(r,:) = GBTresFtempIA;
resBlockGBTA=GBTresF1IIA+resBlockGBTA;
end


else
rb=size(dd1,1);
cb=size(dd1,2);
for r=1:1:rb
     d1=dd1(r,:);
     gbtempI = d1;
if r==1
    G2IDA=V2GBTA1;
end
if r==2
    G2IDA=V2GBTA2;
end
if r==3
    G2IDA=V2GBTA3;
end
if r==4
    G2IDA=V2GBTA4;
end
if r==5
    G2IDA=V2GBTA5;
end
if r==6
    G2IDA=V2GBTA6;
end
if r==7
    G2IDA=V2GBTA7;
end
if r==8
    G2IDA=V2GBTA8;
end 

resgbta1=(G2IDA)*gbtempI';          
GBTresFHIA = resgbta1';     
GBTresFH1IA=zeros(rb,cb);
GBTresFIA(r,:) = GBTresFHIA;
GBTresFIA=GBTresFH1IA+GBTresFIA;
end

GBTresFIA=GBTresFIA';
bb1=GBTresFIA;
rb=size(bb1,1);
cb=size(bb1,2);
for r=1:1:rb
     c1=bb1(r,:);
     ctempI = c1;
if r==1
    G1IDA=V1GBTA1;
end
if r==2
    G1IDA=V1GBTA2;
end
if r==3
    G1IDA=V1GBTA3;
end
if r==4
    G1IDA=V1GBTA4;
end
if r==5
    G1IDA=V1GBTA5;
end
if r==6
    G1IDA=V1GBTA6;
end
if r==7
    G1IDA=V1GBTA7;
end
if r==8
    G1IDA=V1GBTA8;
end
     
resgbta2=ctempI*(G1IDA)';
GBTresFtempIA=resgbta2';
GBTresF1IIA=zeros(rb,cb);
resBlockGBTA(r,:) = GBTresFtempIA;
resBlockGBTA=GBTresF1IIA+resBlockGBTA;
end
resBlockGBTA=resBlockGBTA';
end            
mses(rBlock,cBlock).MSEGBTadptFA(nCoeff+1) = sum(sum((resBlock-resBlockGBTA).^2))/maxMSE; 
 
     
      
      
       
        if energies(rBlock,cBlock).PEGGBTadptFA(nCoeff+1)>1
           energies(rBlock,cBlock).PEGGBTadptFA(nCoeff+1)=1;
        end
        
        end
        energies(rBlock,cBlock).PEGGBTadptFA=smooth_PE(energies(rBlock,cBlock).PEGGBTadptFA);
        mses(rBlock,cBlock).MSEGBTadptFA=smooth_MSE(mses(rBlock,cBlock).MSEGBTadptFA);
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        %Graph-based - actual
        
        im=yr;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEGr(nCoeff+1) = eTmpBlock/eTotal;
   
            ytr = reshape(tmpBlock,rb*cb,1);
                
            resLTGr=(Vr)*ytr;
            
            resLTGr = reshape(resLTGr,rb,cb);
          %  resLTGr=round(resLTGr);
            
            mses(rBlock,cBlock).MSEGr(nCoeff+1)= sum(sum((resBlock-resLTGr).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEGr(nCoeff+1)>1
                energies(rBlock,cBlock).PEGr(nCoeff+1)=1;
         %       mses(rBlock,cBlock).MSEGr(nCoeff+1)=NaN;%MSEGr(nCoeff);
            end
        end
        
        energies(rBlock,cBlock).PEGr=smooth_PE(energies(rBlock,cBlock).PEGr);
        mses(rBlock,cBlock).MSEGr=smooth_MSE(mses(rBlock,cBlock).MSEGr);
        
        
        
         % NN Graph-based - actual
        
        im=yrNN;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEGrNN(nCoeff+1) = eTmpBlock/eTotal;
   
            ytrNN = reshape(tmpBlock,rb*cb,1);
                
            resLTGrNN=(NNV)*ytrNN;
            
            resLTGrNN = reshape(resLTGrNN,rb,cb);
          %  resLTGr=round(resLTGr);
            
            mses(rBlock,cBlock).MSEGrNN(nCoeff+1)= sum(sum((resBlock-resLTGrNN).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEGrNN(nCoeff+1)>1
                energies(rBlock,cBlock).PEGrNN(nCoeff+1)=1;
         %       mses(rBlock,cBlock).MSEGr(nCoeff+1)=NaN;%MSEGr(nCoeff);
            end
        end
        
        energies(rBlock,cBlock).PEGrNN=smooth_PE(energies(rBlock,cBlock).PEGrNN);
        mses(rBlock,cBlock).MSEGrNN=smooth_MSE(mses(rBlock,cBlock).MSEGrNN);
        
        
        
          % NN Graph-based - predicted
        
        im=yrNN_pred;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEGrNN_pred(nCoeff+1) = eTmpBlock/eTotal;
   
            ytrNN_pred = reshape(tmpBlock,rb*cb,1);
                
            resLTGrNN_pred=(NNV_pred)*ytrNN_pred;
            
            resLTGrNN_pred = reshape(resLTGrNN_pred,rb,cb);
          %  resLTGr=round(resLTGr);
            
            mses(rBlock,cBlock).MSEGrNN_pred(nCoeff+1)= sum(sum((resBlock-resLTGrNN_pred).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEGrNN_pred(nCoeff+1)>1
                energies(rBlock,cBlock).PEGrNN_pred(nCoeff+1)=1;
         %       mses(rBlock,cBlock).MSEGr(nCoeff+1)=NaN;%MSEGr(nCoeff);
            end
        end
        
        energies(rBlock,cBlock).PEGrNN_pred=smooth_PE(energies(rBlock,cBlock).PEGrNN_pred);
        mses(rBlock,cBlock).MSEGrNN_pred=smooth_MSE(mses(rBlock,cBlock).MSEGrNN_pred);
        
 
  %Graph-based - actual residial NN 
        
        im=yr_2D;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEGr_2D(nCoeff+1) = eTmpBlock/eTotal;
   
            ytr_2D = reshape(tmpBlock,rb*cb,1);
                
            resLTGr_2D=(Vr_2D)*ytr_2D;
            
            resLTGr_2D = reshape(resLTGr_2D,rb,cb);
          %  resLTGr=round(resLTGr);
            
            mses(rBlock,cBlock).MSEGr_2D(nCoeff+1)= sum(sum((resBlock-resLTGr_2D).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEGr_2D(nCoeff+1)>1
                energies(rBlock,cBlock).PEGr_2D(nCoeff+1)=1;
         %       mses(rBlock,cBlock).MSEGr(nCoeff+1)=NaN;%MSEGr(nCoeff);
            end
        end
        
        energies(rBlock,cBlock).PEGr_2D=smooth_PE(energies(rBlock,cBlock).PEGr_2D);
        mses(rBlock,cBlock).MSEGr_2D=smooth_MSE(mses(rBlock,cBlock).MSEGr_2D);
        
    %Graph-based - actual 2D GBT self loop      
        im=yr_2D_old;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEGr_2D_old(nCoeff+1) = eTmpBlock/eTotal;
   
            ytr_2D_old = reshape(tmpBlock,rb*cb,1);
                
            resLTGr_2D_old=(Vr_2D_old)*ytr_2D_old;
            
            resLTGr_2D_old = reshape(resLTGr_2D_old,rb,cb);
          %  resLTGr=round(resLTGr);
            
            mses(rBlock,cBlock).MSEGr_2D_old(nCoeff+1)= sum(sum((resBlock-resLTGr_2D_old).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEGr_2D_old(nCoeff+1)>1
                energies(rBlock,cBlock).PEGr_2D_old(nCoeff+1)=1;
         %       mses(rBlock,cBlock).MSEGr(nCoeff+1)=NaN;%MSEGr(nCoeff);
            end
        end
        
        energies(rBlock,cBlock).PEGr_2D_old=smooth_PE(energies(rBlock,cBlock).PEGr_2D_old);
        mses(rBlock,cBlock).MSEGr_2D_old=smooth_MSE(mses(rBlock,cBlock).MSEGr_2D_old);
        
         
        
        
        
        
        
        
        
      %Graph-based - actual separable
        
        im=GBTadptFinal;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            gbttmpBlock=mask.*im;
            gbteTmpBlock=sum(sum(gbttmpBlock.^2));
            eTmpBlockG=gbteTmpBlock;
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEGGBTadptF(nCoeff+1) = gbteTmpBlock/eTotal;
            
            
dd1=gbttmpBlock;   
if mode < 18
dd1=dd1';
rb=size(dd1,1);
cb=size(dd1,2);
for r=1:1:rb
     d1=dd1(r,:);
%     %eval(['DCTI' num2str(r) ' =d1']);
     gbtempI = d1;
     %  d=dct2(c);% DCT Calculation
if r==1
    G2ID=V2GBT1;
end
if r==2
    G2ID=V2GBT2;
end
if r==3
    G2ID=V2GBT3;
end
if r==4
    G2ID=V2GBT4;
end
if r==5
    G2ID=V2GBT5;
end
if r==6
    G2ID=V2GBT6;
end
if r==7
    G2ID=V2GBT7;
end
if r==8
    G2ID=V2GBT8;
end 

resgbt1=(G2ID)*gbtempI';        
GBTresFHI = resgbt1';     
GBTresFH1I=zeros(rb,cb);
GBTresFI(r,:) = GBTresFHI;
GBTresFI=GBTresFH1I+GBTresFI;
end 
GBTresFI=GBTresFI';

bb1=GBTresFI;
rb=size(dd1,1);
cb=size(bb1,2);
for r=1:1:rb
     c1=bb1(r,:);
     ctempI = c1;
if r==1
    G1ID=V1GBT1;
end
if r==2
    G1ID=V1GBT2;
end
if r==3
    G1ID=V1GBT3;
end
if r==4
    G1ID=V1GBT4;
end
if r==5
    G1ID=V1GBT5;
end
if r==6
    G1ID=V1GBT6;
end
if r==7
    G1ID=V1GBT7;
end
if r==8
    G1ID=V1GBT8;
end
     
resgbt2=ctempI*(G1ID)';
GBTresFtempI=resgbt2';
GBTresF1II=zeros(rb,cb);
resBlockGBT(r,:) = GBTresFtempI;
resBlockGBT=GBTresF1II+resBlockGBT;
end

    
else
rb=size(dd1,1);
cb=size(dd1,2);
for r=1:1:rb
     d1=dd1(r,:);
%     %eval(['DCTI' num2str(r) ' =d1']);
     gbtempI = d1;
     %  d=dct2(c);% DCT Calculation
if r==1
    G2ID=V2GBT1;
end
if r==2
    G2ID=V2GBT2;
end
if r==3
    G2ID=V2GBT3;
end
if r==4
    G2ID=V2GBT4;
end
if r==5
    G2ID=V2GBT5;
end
if r==6
    G2ID=V2GBT6;
end
if r==7
    G2ID=V2GBT7;
end
if r==8
    G2ID=V2GBT8;
end 

resgbt1=(G2ID)*gbtempI';         
GBTresFHI = resgbt1';     
GBTresFH1I=zeros(rb,cb);
GBTresFI(r,:) = GBTresFHI;
GBTresFI=GBTresFH1I+GBTresFI;
end

GBTresFI=GBTresFI';
bb1=GBTresFI;
rb=size(bb1,1);
cb=size(bb1,2);
for r=1:1:rb
     c1=bb1(r,:);
     ctempI = c1;
if r==1
    G1ID=V1GBT1;
end
if r==2
    G1ID=V1GBT2;
end
if r==3
    G1ID=V1GBT3;
end
if r==4
    G1ID=V1GBT4;
end
if r==5
    G1ID=V1GBT5;
end
if r==6
    G1ID=V1GBT6;
end
if r==7
    G1ID=V1GBT7;
end
if r==8
    G1ID=V1GBT8;
end
     
resgbt2=ctempI*(G1ID)';
GBTresFtempI=resgbt2';
GBTresF1II=zeros(rb,cb);
resBlockGBT(r,:) = GBTresFtempI;
resBlockGBT=GBTresF1II+resBlockGBT;
end
resBlockGBT=resBlockGBT';
end
            
mses(rBlock,cBlock).MSEGBTadptF(nCoeff+1) = sum(sum((resBlock-resBlockGBT).^2))/maxMSE;
                        
       
        if energies(rBlock,cBlock).PEGGBTadptF(nCoeff+1)>1
           energies(rBlock,cBlock).PEGGBTadptF(nCoeff+1)=1;
        end
        
        end
        energies(rBlock,cBlock).PEGGBTadptF=smooth_PE(energies(rBlock,cBlock).PEGGBTadptF);
        mses(rBlock,cBlock).MSEGBTadptF=smooth_MSE(mses(rBlock,cBlock).MSEGBTadptF);        
        
        
        
        
        
        
        
        
        
    %{    
                %Graph-based - actual with 8 connectivity
        
        im=yrac;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEGrA(nCoeff+1) = eTmpBlock/eTotal;
   
            ytrA = reshape(tmpBlock,rb*cb,1);
                
            resLTGrA=(Vac)*ytrA;
            
            resLTGrA = reshape(resLTGrA,rb,cb);
            resLTGrA=round(resLTGrA);
            
            mses(rBlock,cBlock).MSEGrA(nCoeff+1)= sum(sum((resBlock-resLTGrA).^2))/maxMSE;
            
            if energies(rBlock,cBlock).PEGrA(nCoeff+1)>1
                energies(rBlock,cBlock).PEGrA(nCoeff+1)=NaN;
                mses(rBlock,cBlock).MSEGrA(nCoeff+1)=NaN;%MSEGrA(nCoeff);
            end
        end
        
     %}   
        
        
        
        
        
        
        
        
        
        %DST coefficients
        
%        im=DSTres;
%         maxCoeff=max(max(abs(im)));
%         minCoeff=min(min(abs(im)));
%         for th=maxCoeff:-step:minCoeff-step
%             
%             mask=abs(im)>th;
%             tmpBlock=mask.*im;
%             eTmpBlock=sum(sum(tmpBlock.^2));
%             nCoeff=sum(sum(mask));
%            
%             energies(rBlock,cBlock).PEDST(nCoeff+1) = eTmpBlock/eTotal;
%             
%             
%             resLT2=idst(idst(tmpBlock')');
%             
%             mses(rBlock,cBlock).MSEDST(nCoeff+1) = sum(sum((resBlock-resLT2).^2))/maxMSE;
%             
%             if energies(rBlock,cBlock).PEDST(nCoeff+1)>1
%                 energies(rBlock,cBlock).PEDST(nCoeff+1)=NaN;
%                 mses(rBlock,cBlock).MSEDST(nCoeff+1)=NaN;%MSEDST(nCoeff);
%             end
%         end
        
        %ASI coeffcients
        im=ASIres;
        maxCoeff=max(max(abs(im)));
        minCoeff=min(min(abs(im)));
        for th=maxCoeff:-step:minCoeff-step
            
            mask=abs(im)>th;
            tmpBlock=mask.*im;
            eTmpBlock=sum(sum(tmpBlock.^2));
            nCoeff=sum(sum(mask));
            
            energies(rBlock,cBlock).PEASI(nCoeff+1) = eTmpBlock/eTotal;

            if (mode>=0 && mode<=10) || (mode>=26 && mode<=34)
                
                for col=1:1:cb
                    resLT3t(:,col)=tmpBlock(:,col)'/DST7; %the DST is applied to each col
                end
                resLT3=idct(resLT3t');
                resLT3=resLT3';
                
                
            end
            
            if mode>=11 && mode<=25
                for row=1:1:rb
                    resLT3t(row,:)=tmpBlock(row,:)/DST7; %the DST is applied to each col
                end
                
                for col=1:1:cb
                    resLT3(:,col)=resLT3t(:,col)'/DST7; %the DST is applied to each col
                end
            end
            
            mses(rBlock,cBlock).MSEASI(nCoeff+1) = sum(sum((resBlock-resLT3).^2))/maxMSE;
            
%uncomment this when energy of transform coefficents may be too large for
%a certain budget
             if energies(rBlock,cBlock).PEASI(nCoeff+1)>1
                 energies(rBlock,cBlock).PEASI(nCoeff+1)=1;
%                 mses(rBlock,cBlock).MSEASI(nCoeff+1)=NaN;%MSEASI(nCoeff);
             end
         
        end
        energies(rBlock,cBlock).PEASI=smooth_PE(energies(rBlock,cBlock).PEASI);
        mses(rBlock,cBlock).MSEASI=smooth_MSE(mses(rBlock,cBlock).MSEASI);
        
        aa=aa+1;
   end
end
%%%%%%%%%%Block_Adj3dgt_3D=cat(3,Adj3dgt1,Adj3dgt2,Adj3dgt3,Adj3dgt4);
%%%%%%%%%%Block_Ref_3D_Comb=cat(3,Block_Ref_3D1,Block_Ref_3D2,Block_Ref_3D3,Block_Ref_3D4);
%%%%%%%%%%ResBlock_Ref_3D_Comb=cat(3,ResBlock_Ref_3D1,ResBlock_Ref_3D2,ResBlock_Ref_3D3,ResBlock_Ref_3D4);
%{
ZZ=zeros(1,aa);
for count1=1:1:aa
    kk=eval(['Adj3dgt' num2str(count1)]);
    ZZ(count1) = kk;
    Block_Adj3dgt_3D=cat(3,eval(['Adj3dgt' num2str(count1)]));
    Block_Ref_3D_Comb=cat(3,eval(['Block_Ref_3D' num2str(count1)]));
    ResBlock_Ref_3D_Comb=cat(3,eval(['ResBlock_Ref_3D' num2str(count1)]));
    Block_Adj3dgt_3D=(Block_Adj3dgt_3D);
  %  eval(['Adj3dgt' num2str(count-1) ' =Adj3dgt']);
end
%}

Block_Adj3dgt_3D=cat(3,Adj3dgt1,Adj3dgt2,Adj3dgt3,Adj3dgt4,Adj3dgt5,Adj3dgt6,Adj3dgt7,Adj3dgt8,Adj3dgt9,Adj3dgt10,Adj3dgt11,Adj3dgt12,Adj3dgt13,Adj3dgt14,Adj3dgt15,Adj3dgt16,Adj3dgt17,Adj3dgt18,Adj3dgt19,Adj3dgt20,...
Adj3dgt21,Adj3dgt22,Adj3dgt23,Adj3dgt24,Adj3dgt25,Adj3dgt26,Adj3dgt27,Adj3dgt28,Adj3dgt29,Adj3dgt30,Adj3dgt31,Adj3dgt32,Adj3dgt33,Adj3dgt34,Adj3dgt35,Adj3dgt36,Adj3dgt37,Adj3dgt38,Adj3dgt39,Adj3dgt40,...
Adj3dgt41,Adj3dgt42,Adj3dgt43,Adj3dgt44,Adj3dgt45,Adj3dgt46,Adj3dgt47,Adj3dgt48,Adj3dgt49,Adj3dgt50,Adj3dgt51,Adj3dgt52,Adj3dgt53,Adj3dgt54,Adj3dgt55,Adj3dgt56,Adj3dgt57,Adj3dgt58,Adj3dgt59,Adj3dgt60,Adj3dgt61,Adj3dgt62,Adj3dgt63,Adj3dgt64);
%%%%%%%%%Block_Adj3dgt_3D=cat(3,Adj3dgt1,Adj3dgt2,Adj3dgt3,Adj3dgt4);

Block_Ref_3D_Comb=cat(3,Block_Ref_3D1,Block_Ref_3D2,Block_Ref_3D3,Block_Ref_3D4,Block_Ref_3D5,Block_Ref_3D6,Block_Ref_3D7,Block_Ref_3D8,Block_Ref_3D9,Block_Ref_3D10,Block_Ref_3D11,Block_Ref_3D12,Block_Ref_3D13,Block_Ref_3D14,Block_Ref_3D15,Block_Ref_3D16,Block_Ref_3D17,Block_Ref_3D18,Block_Ref_3D19,Block_Ref_3D20,...
Block_Ref_3D21,Block_Ref_3D22,Block_Ref_3D23,Block_Ref_3D24,Block_Ref_3D25,Block_Ref_3D26,Block_Ref_3D27,Block_Ref_3D28,Block_Ref_3D29,Block_Ref_3D30,...
Block_Ref_3D31,Block_Ref_3D32,Block_Ref_3D33,Block_Ref_3D34,Block_Ref_3D35,Block_Ref_3D36,Block_Ref_3D37,Block_Ref_3D38,Block_Ref_3D39,Block_Ref_3D40,...
Block_Ref_3D41,Block_Ref_3D42,Block_Ref_3D43,Block_Ref_3D44,Block_Ref_3D45,Block_Ref_3D46,Block_Ref_3D47,Block_Ref_3D48,Block_Ref_3D49,Block_Ref_3D50,...
Block_Ref_3D51,Block_Ref_3D52,Block_Ref_3D53,Block_Ref_3D54,Block_Ref_3D55,Block_Ref_3D56,Block_Ref_3D57,Block_Ref_3D58,Block_Ref_3D59,Block_Ref_3D60,Block_Ref_3D61,Block_Ref_3D62,Block_Ref_3D63,Block_Ref_3D64);
%ResBlock_Ref_3D_Comb=cat(3,ResBlock_Ref_3D1,ResBlock_Ref_3D2,ResBlock_Ref_3D3,ResBlock_Ref_3D4,ResBlock_Ref_3D5,ResBlock_Ref_3D6,ResBlock_Ref_3D7,ResBlock_Ref_3D8,ResBlock_Ref_3D9,ResBlock_Ref_3D10,ResBlock_Ref_3D11,ResBlock_Ref_3D12,ResBlock_Ref_3D13,ResBlock_Ref_3D14,ResBlock_Ref_3D15,ResBlock_Ref_3D16);
%%%%%%%Block_Ref_3D_Comb=cat(3,Block_Ref_3D1,Block_Ref_3D2,Block_Ref_3D3,Block_Ref_3D4);


 %%%%%----------------------------

%PLOT THE OVERALL PRESERVED ENERGY AND MSE
% THIS IS PLOT  IN TERMS OF THE ENERGY COMPACTION PERFORMANCE
% SEE PAPER: Transforms for Intra Prediction Residuals Based on Prediction Inaccuracy Modeling
% BY Xun Cai AND Jae S. Lim
% IN GENERAL THE PLOTS SHOW THE PERCENTAGE OF PRESERVED COEFFICIENTS VS.
% THE PERCENTAGE OF PRESERVED ENERGY

totalPEDCT=zeros(rb*cb+1,1);
totalMSEDCT=zeros(rb*cb+1,1);
totalPEASI=zeros(rb*cb+1,1);
totalMSEASI=zeros(rb*cb+1,1);
%totalPEDST=zeros(rb*cb,1);
%totalMSEDST=zeros(rb*cb,1);
totalPEKLT=zeros(rb*cb+1,1);
totalMSEKLT=zeros(rb*cb+1,1);

totalPEG=zeros(rb*cb+1,1);
totalMSEG=zeros(rb*cb+1,1);

totalPEG_ORT=zeros(rb*cb+1,1);
totalMSEG_ORT=zeros(rb*cb+1,1);

totalPEG_2D=zeros(rb*cb+1,1);
totalMSEG_2D=zeros(rb*cb+1,1);

totalPEG_NLM=zeros(rb*cb+1,1);
totalMSEG_NLM=zeros(rb*cb+1,1);

totalPEG_NLM_2D=zeros(rb*cb+1,1);
totalMSEG_NLM_2D=zeros(rb*cb+1,1);

totalPEG_TMB=zeros(rb*cb+1,1);
totalMSEG_TMB=zeros(rb*cb+1,1);

totalPEG_TMB_2D=zeros(rb*cb+1,1);
totalMSEG_TMB_2D=zeros(rb*cb+1,1);


totalPEG_TMBNLM=zeros(rb*cb+1,1);
totalMSEG_TMBNLM=zeros(rb*cb+1,1);

totalPEG_TMBNLM_2D=zeros(rb*cb+1,1);
totalMSEG_TMBNLM_2D=zeros(rb*cb+1,1);


totalPEG_TMBNLM_2D_old=zeros(rb*cb+1,1);
totalMSEG_TMBNLM_2D_old=zeros(rb*cb+1,1);

totalPEG_PI=zeros(rb*cb+1,1);
totalMSEG_PI=zeros(rb*cb+1,1);

totalPEG_PI_2D=zeros(rb*cb+1,1);
totalMSEG_PI_2D=zeros(rb*cb+1,1);


totalPEGGBTadptFA=zeros(rberes*cberes+1,1);
totalMSEGBTadptFA=zeros(rb*cb+1,1);
totalPEGGBTadptF=zeros(rb*cb+1,1);
totalMSEGBTadptF=zeros(rb*cb+1,1);
totalPEGTMFA=zeros(rberes*cberes+1,1);
totalMSETMFA=zeros(rb*cb+1,1);
totalPEGTMFB=zeros(rberes*cberes+1,1);
totalMSETMFB=zeros(rb*cb+1,1);
%{
totalPEGE=zeros(rb*cb,1);
totalMSEGE=zeros(rb*cb,1);
%}
totalPEGr=zeros(rb*cb+1,1);
totalMSEGr=zeros(rb*cb+1,1);

totalPEGrNN=zeros(rb*cb+1,1);
totalMSEGrNN=zeros(rb*cb+1,1);

totalPEGrNN_pred=zeros(rb*cb+1,1);
totalMSEGrNN_pred=zeros(rb*cb+1,1);

totalPEGr_2D=zeros(rb*cb+1,1);
totalMSEGr_2D=zeros(rb*cb+1,1);

totalPEGr_2D_old=zeros(rb*cb+1,1);
totalMSEGr_2D_old=zeros(rb*cb+1,1);
%{
totalPEGrA=zeros(rb*cb,1);
totalMSEGrA=zeros(rb*cb,1);
%}
%totalValuesPE=zeros(rb*cb,1);
%totalValuesMSE=zeros(rb*cb,1);

% totValuesPEDCT=zeros(rb*cb,1);
% totValuesPEKLT=zeros(rb*cb,1);
% totValuesPEASI=zeros(rb*cb,1);
% totValuesPEGr=zeros(rb*cb,1);
% totValuesPEG=zeros(rb*cb,1);



totalPEGBTONLLap=zeros(rb*cb+1,1);
totalMSEGBTONLLap=zeros(rb*cb+1,1);

%{
totValuesPEGrA=zeros(rb*cb,1);
totValuesPEGE=zeros(rb*cb,1);
%}
for row=1:1:rBlock
    for col=1:1:cBlock
        
        totValuesPEDCT=0;
        totValuesPEKLT=0;
        totValuesPEASI=0;
        totValuesPEGr=0;
         totValuesPEORT=0;
         totValuesPEGrNN=0;
          totValuesPEGrNN_pred=0;
        totValuesPEGr_2D=0;
         totValuesPEGr_2D_old=0;
        totValuesPEG=0;
         totValuesPEG_2D=0;
        totValuesPEG_NLM=0;
          totValuesPEG_NLM_2D=0;
        totValuesPEG_TMB=0;
         totValuesPEG_TMB_2D=0;
        totValuesPEG_TMBNLM=0;
         totValuesPEG_TMBNLM_2D=0;
          totValuesPEG_TMBNLM_2D_old=0;
        totValuesPEG_PI=0;
         totValuesPEG_PI_2D=0;
        totValuesPEGGBTadptFA=0;
        totValuesPEGGBTadptF=0;
        totValuesPEGTMFA=0;
        totValuesPEGTMFB=0;
         totValuesPEGBTONLLap=0;
        for coeff=2:1:rb*cb+1
            if energies(row,col).PEGBTONLLap(coeff)>0
                totValuesPEGBTONLLap= totValuesPEGBTONLLap+1;
            end
            if energies(row,col).PEDCT(coeff)>0
                totValuesPEDCT= totValuesPEDCT+1;
            end
            if energies(row,col).PEKLT(coeff)>0
                totValuesPEKLT= totValuesPEKLT+1;
            end
             if energies(row,col).PEG_ORT(coeff)>0
                 totValuesPEORT=  totValuesPEORT+1;
            end
            if energies(row,col).PEASI(coeff)>0
                totValuesPEASI= totValuesPEASI+1;
            end
            if energies(row,col).PEG(coeff)>0
                totValuesPEG= totValuesPEG+1;
            end
             if energies(row,col).PEG_2D(coeff)>0
                totValuesPEG_2D= totValuesPEG_2D+1;
            end
            
             if energies(row,col).PEG_NLM(coeff)>0
                totValuesPEG_NLM= totValuesPEG_NLM+1;
             end
             if energies(row,col).PEG_NLM_2D(coeff)>0
                totValuesPEG_NLM_2D= totValuesPEG_NLM_2D+1;
            end
            
            if energies(row,col).PEG_TMB(coeff)>0
                totValuesPEG_TMB= totValuesPEG_TMB+1;
            end
             if energies(row,col).PEG_TMB_2D(coeff)>0
                totValuesPEG_TMB_2D= totValuesPEG_TMB_2D+1;
            end
            if energies(row,col).PEG_TMBnlm(coeff)>0
                totValuesPEG_TMBNLM= totValuesPEG_TMBNLM+1;
            end    
             if energies(row,col).PEG_TMBnlm_2D(coeff)>0
                totValuesPEG_TMBNLM_2D= totValuesPEG_TMBNLM_2D+1;
             end 
             if energies(row,col).PEG_TMBnlm_2D_old(coeff)>0
                totValuesPEG_TMBNLM_2D_old= totValuesPEG_TMBNLM_2D_old+1;
            end 
            if energies(row,col).PEG_PI(coeff)>0
                totValuesPEG_PI= totValuesPEG_PI+1;
            end
            if energies(row,col).PEG_PI_2D(coeff)>0
                totValuesPEG_PI_2D= totValuesPEG_PI_2D+1;
            end
            if energies(row,col).PEGr(coeff)>0
                totValuesPEGr= totValuesPEGr+1;
            end
            
            if energies(row,col).PEGrNN(coeff)>0
                totValuesPEGrNN= totValuesPEGrNN+1;
            end
            
             if energies(row,col).PEGrNN_pred(coeff)>0
                totValuesPEGrNN_pred= totValuesPEGrNN_pred+1;
            end
             if energies(row,col).PEGr_2D(coeff)>0
                totValuesPEGr_2D= totValuesPEGr_2D+1;
             end
              if energies(row,col).PEGr_2D_old(coeff)>0
                totValuesPEGr_2D_old= totValuesPEGr_2D_old+1;
            end
            if energies(row,col).PEGGBTadptFA(coeff)>0
                totValuesPEGGBTadptFA= totValuesPEGGBTadptFA+1;
            end
            if energies(row,col).PEGGBTadptF(coeff)>0
                totValuesPEGGBTadptF= totValuesPEGGBTadptF+1;
            end
            if energies(row,col).PEGTMFA(coeff)>0
                totValuesPEGTMFA= totValuesPEGTMFA+1;
            end
            if energies(row,col).PEGTMFB(coeff)>0
                totValuesPEGTMFB= totValuesPEGTMFB+1;
            end
            %{
            if energies(row,col).PEGE(coeff)>0
                totValuesPEGE(coeff)= totValuesPEGE(coeff)+1;
            end
            if energies(row,col).PEGrA(coeff)>0
                totValuesPEGrA(coeff)= totValuesPEGrA(coeff)+1;
            end
            %}
            
            totalPEDCT(coeff)=totalPEDCT(coeff)+energies(row,col).PEDCT(coeff);
            %totalPEDST(coeff)=totalPEDST(coeff)+energies(row,col).PEDST(coeff);
            totalPEKLT(coeff)=totalPEKLT(coeff)+energies(row,col).PEKLT(coeff);
            totalPEASI(coeff)=totalPEASI(coeff)+energies(row,col).PEASI(coeff);
            totalPEGBTONLLap(coeff)=totalPEGBTONLLap(coeff)+energies(row,col).PEGBTONLLap(coeff);
%             if ~isnan(energies(row,col).PEASI(coeff))
%                 %totalPEASI(coeff)=totalPEASI(coeff)+0;
%             %else
%                totalPEASI(coeff)=totalPEASI(coeff)+energies(row,col).PEASI(coeff);
%                 totalValuesPE(coeff)=totalValuesPE(coeff)+1;
%             end
            
            totalPEG(coeff)=totalPEG(coeff)+energies(row,col).PEG(coeff);
             totalPEG_ORT(coeff)=totalPEG_ORT(coeff)+energies(row,col).PEG_ORT(coeff);
            totalPEG_2D(coeff)=totalPEG_2D(coeff)+energies(row,col).PEG_2D(coeff);
            totalPEG_NLM(coeff)=totalPEG_NLM(coeff)+energies(row,col).PEG_NLM(coeff);
              totalPEG_NLM_2D(coeff)=totalPEG_NLM_2D(coeff)+energies(row,col).PEG_NLM_2D(coeff);
              totalPEG_TMB(coeff)=totalPEG_TMB(coeff)+energies(row,col).PEG_TMB(coeff);
              totalPEG_TMBNLM(coeff)=totalPEG_TMBNLM(coeff)+energies(row,col).PEG_TMBnlm(coeff);
             totalPEG_PI(coeff)=totalPEG_PI(coeff)+energies(row,col).PEG_PI(coeff);
             
              totalPEG_TMB_2D(coeff)=totalPEG_TMB_2D(coeff)+energies(row,col).PEG_TMB_2D(coeff);
              totalPEG_TMBNLM_2D(coeff)=totalPEG_TMBNLM_2D(coeff)+energies(row,col).PEG_TMBnlm_2D(coeff);
              
              totalPEG_TMBNLM_2D_old(coeff)=totalPEG_TMBNLM_2D_old(coeff)+energies(row,col).PEG_TMBnlm_2D_old(coeff);
             totalPEG_PI_2D(coeff)=totalPEG_PI_2D(coeff)+energies(row,col).PEG_PI_2D(coeff);
            totalPEGr(coeff)=totalPEGr(coeff)+energies(row,col).PEGr(coeff);
            
             totalPEGrNN(coeff)=totalPEGrNN(coeff)+energies(row,col).PEGrNN(coeff);
             
                totalPEGrNN_pred(coeff)=totalPEGrNN_pred(coeff)+energies(row,col).PEGrNN_pred(coeff);
             totalPEGr_2D(coeff)=totalPEGr_2D(coeff)+energies(row,col).PEGr_2D(coeff);
               totalPEGr_2D_old(coeff)=totalPEGr_2D_old(coeff)+energies(row,col).PEGr_2D_old(coeff);
            totalPEGGBTadptFA(coeff)=totalPEGGBTadptFA(coeff)+energies(row,col).PEGGBTadptFA(coeff);
            totalPEGGBTadptF(coeff)=totalPEGGBTadptF(coeff)+energies(row,col).PEGGBTadptF(coeff);
            totalPEGTMFA(coeff)=totalPEGTMFA(coeff)+energies(row,col).PEGTMFA(coeff);
            totalPEGTMFB(coeff)=totalPEGTMFB(coeff)+energies(row,col).PEGTMFB(coeff);
         %{   
            totalPEGE(coeff)=totalPEGE(coeff)+energies(row,col).PEGE(coeff);
            totalPEGrA(coeff)=totalPEGrA(coeff)+energies(row,col).PEGrA(coeff);
          %}  
            
          totalMSEGBTONLLap(coeff)=totalMSEGBTONLLap(coeff)+mses(row,col).MSEGBTONLLap(coeff);
          totalMSEASI(coeff)=totalMSEASI(coeff)+mses(row,col).MSEASI(coeff);
            
            totalMSEDCT(coeff)=totalMSEDCT(coeff)+mses(row,col).MSEDCT(coeff);
            %totalMSEDST(coeff)=totalMSEDST(coeff)+mses(row,col).MEDST(coeff);
            totalMSEKLT(coeff)=totalMSEKLT(coeff)+mses(row,col).MSEKLT(coeff);

%             if ~isnan(mses(row,col).MSEASI(coeff))
%                 totalMSEASI(coeff)=totalMSEASI(coeff)+mses(row,col).MSEASI(coeff);
%                totalValuesMSE(coeff)=totalValuesMSE(coeff)+1;
%             end
            
            totalMSEG(coeff)=totalMSEG(coeff)+mses(row,col).MSEG(coeff);
            totalMSEG_ORT(coeff)=totalMSEG_ORT(coeff)+mses(row,col).MSEG_ORT(coeff);
          
             totalMSEG_NLM(coeff)=totalMSEG_NLM(coeff)+mses(row,col).MSEG_NLM(coeff);
             
             totalMSEG_TMB(coeff)=totalMSEG_TMB(coeff)+mses(row,col).MSEG_TMB(coeff);
             totalMSEG_TMBNLM(coeff)=totalMSEG_TMBNLM(coeff)+mses(row,col).MSEG_TMBnlm(coeff);
             totalMSEG_PI(coeff)=totalMSEG_PI(coeff)+mses(row,col).MSEG_PI(coeff);
             
             totalMSEG_2D(coeff)=totalMSEG_2D(coeff)+mses(row,col).MSEG_2D(coeff);
            %  totalMSEG_2D_old(coeff)=totalMSEG_2D_old(coeff)+mses(row,col).MSEG_2D_old(coeff);
             totalMSEG_NLM_2D(coeff)=totalMSEG_NLM_2D(coeff)+mses(row,col).MSEG_NLM_2D(coeff);
             
             totalMSEG_TMB_2D(coeff)=totalMSEG_TMB_2D(coeff)+mses(row,col).MSEG_TMB_2D(coeff);
             totalMSEG_TMBNLM_2D(coeff)=totalMSEG_TMBNLM_2D(coeff)+mses(row,col).MSEG_TMBnlm_2D(coeff);
               totalMSEG_TMBNLM_2D_old(coeff)=totalMSEG_TMBNLM_2D_old(coeff)+mses(row,col).MSEG_TMBnlm_2D_old(coeff);
             totalMSEG_PI_2D(coeff)=totalMSEG_PI_2D(coeff)+mses(row,col).MSEG_PI_2D(coeff);
             
            totalMSEGr(coeff)=totalMSEGr(coeff)+mses(row,col).MSEGr(coeff);
            
             totalMSEGrNN(coeff)=totalMSEGrNN(coeff)+mses(row,col).MSEGrNN(coeff);
             
              totalMSEGrNN_pred(coeff)=totalMSEGrNN_pred(coeff)+mses(row,col).MSEGrNN_pred(coeff);
             totalMSEGr_2D(coeff)=totalMSEGr_2D(coeff)+mses(row,col).MSEGr_2D(coeff);
             
            totalMSEGr_2D_old(coeff)=totalMSEGr_2D_old(coeff)+mses(row,col).MSEGr_2D_old(coeff);
            totalMSEGBTadptFA(coeff)=totalMSEGBTadptFA(coeff)+mses(row,col).MSEGBTadptFA(coeff);
            totalMSEGBTadptF(coeff)=totalMSEGBTadptF(coeff)+mses(row,col).MSEGBTadptF(coeff);
            totalMSETMFA(coeff)=totalMSETMFA(coeff)+mses(row,col).MSETMFA(coeff);
            totalMSETMFB(coeff)=totalMSETMFB(coeff)+mses(row,col).MSETMFB(coeff);
           %{ 
            totalMSEGE(coeff)=totalMSEGE(coeff)+mses(row,col).MSEGE(coeff);
            totalMSEGrA(coeff)=totalMSEGrA(coeff)+mses(row,col).MSEGrA(coeff);
            %}
            
            
        end
            if totValuesPEDCT ~= rb*cb
                disp('Warning - totvalues DCT');                
            end
            if totValuesPEKLT ~= rb*cb
                disp('Warning - totvalues KLT');
            end
            if totValuesPEASI ~= rb*cb
                disp('Warning - totvalues ASI');
            end
            if totValuesPEG ~= rb*cb
                disp('Warning - totvalues G');
            end
             if totValuesPEG_2D ~= rb*cb
                disp('Warning - totvalues G_2D');
            end
            
            if totValuesPEG_NLM ~= rb*cb
                disp('Warning - totvalues G_NLM');
            end
             if totValuesPEG_NLM_2D ~= rb*cb
                disp('Warning - totvalues G_NLM_2D');
            end
            
            
             if totValuesPEG_TMB ~= rb*cb
                disp('Warning - totvalues G_TMB');
             end
              if totValuesPEG_TMB_2D ~= rb*cb
                disp('Warning - totvalues G_TMB_2D');
             end
             if totValuesPEG_TMBNLM ~= rb*cb
                disp('Warning - totvalues G_TMB');
             end
             if totValuesPEG_TMBNLM_2D ~= rb*cb
                disp('Warning - totvalues G_TMB_2D');
             end
             if totValuesPEG_TMBNLM_2D_old ~= rb*cb
                disp('Warning - totvalues G_TMB_2D_old');
            end
             if totValuesPEG_PI ~= rb*cb
                disp('Warning - totvalues G_PI');
             end
             if totValuesPEG_PI_2D ~= rb*cb
                disp('Warning - totvalues G_PI_2D');
            end
            
            if totValuesPEGr ~= rb*cb
                disp('Warning - totvalues GR');
            end 
            
             if totValuesPEGrNN ~= rb*cb
                disp('Warning - totvalues GRNN');
             end
            
              if totValuesPEGrNN_pred ~= rb*cb
                disp('Warning - totvalues GRNN_pred');
             end
            
            if totValuesPEGr_2D ~= rb*cb
                disp('Warning - totvalues GR_2D');
            end  
            
              if totValuesPEGr_2D_old ~= rb*cb
                disp('Warning - totvalues GR_2D_old');
            end  
    end
    
end

totalPEDCT=totalPEDCT./(rBlock*cBlock);
totalMSEDCT=totalMSEDCT./(rBlock*cBlock);
%totalPEDST=totalPEDST/(rb*cb);
%totalMSEDST=totalMSEDST/(rb*cb);
totalPEKLT=totalPEKLT./(rBlock*cBlock);
totalMSEKLT=totalMSEKLT./(rBlock*cBlock);

totalPEG=totalPEG./(rBlock*cBlock);
totalMSEG=totalMSEG./(rBlock*cBlock);

totalPEG_ORT=totalPEG_ORT./(rBlock*cBlock);
totalMSEG_ORT=totalMSEG_ORT./(rBlock*cBlock);

totalPEG_2D=totalPEG_2D./(rBlock*cBlock);
totalMSEG_2D=totalMSEG_2D./(rBlock*cBlock);


totalPEG_NLM=totalPEG_NLM./(rBlock*cBlock);
totalMSEG_NLM=totalMSEG_NLM./(rBlock*cBlock);

totalPEG_NLM_2D=totalPEG_NLM_2D./(rBlock*cBlock);
totalMSEG_NLM_2D=totalMSEG_NLM_2D./(rBlock*cBlock);



totalPEG_TMB=totalPEG_TMB./(rBlock*cBlock);
totalMSEG_TMB=totalMSEG_TMB./(rBlock*cBlock);

totalPEG_TMB_2D=totalPEG_TMB_2D./(rBlock*cBlock);
totalMSEG_TMB_2D=totalMSEG_TMB_2D./(rBlock*cBlock);

totalPEG_TMBNLM=totalPEG_TMBNLM./(rBlock*cBlock);
totalMSEG_TMBNLM=totalMSEG_TMBNLM./(rBlock*cBlock);

totalPEG_TMBNLM_2D=totalPEG_TMBNLM_2D./(rBlock*cBlock);
totalMSEG_TMBNLM_2D=totalMSEG_TMBNLM_2D./(rBlock*cBlock);

totalPEG_TMBNLM_2D_old=totalPEG_TMBNLM_2D_old./(rBlock*cBlock);
totalMSEG_TMBNLM_2D_old=totalMSEG_TMBNLM_2D_old./(rBlock*cBlock);


totalPEG_PI=totalPEG_PI./(rBlock*cBlock);
totalMSEG_PI=totalMSEG_PI./(rBlock*cBlock);

totalPEG_PI_2D=totalPEG_PI_2D./(rBlock*cBlock);
totalMSEG_PI_2D=totalMSEG_PI_2D./(rBlock*cBlock);
% 
%{
totalPEGE=totalPEGE./totValuesPEGE;
totalMSEGE=totalMSEGE./totValuesPEGE;
%}
totalPEGr=totalPEGr./(rBlock*cBlock);
totalMSEGr=totalMSEGr./(rBlock*cBlock);

totalPEGrNN=totalPEGrNN./(rBlock*cBlock);
totalMSEGrNN=totalMSEGrNN./(rBlock*cBlock);

totalPEGrNN_pred=totalPEGrNN_pred./(rBlock*cBlock);
totalMSEGrNN_pred=totalMSEGrNN_pred./(rBlock*cBlock);


totalPEGr_2D=totalPEGr_2D./(rBlock*cBlock);
totalMSEGr_2D=totalMSEGr_2D./(rBlock*cBlock);

totalPEGr_2D_old=totalPEGr_2D_old./(rBlock*cBlock);
totalMSEGr_2D_old=totalMSEGr_2D_old./(rBlock*cBlock);
%{
totalPEGrA=totalPEGrA./totValuesPEGrA;
totalMSEGrA=totalMSEGrA./totValuesPEGrA;
%}
totalPEASI=totalPEASI./(rBlock*cBlock);
totalMSEASI=totalMSEASI./(rBlock*cBlock);

totalPEGGBTadptFA=totalPEGGBTadptFA./(rBlock*cBlock);
totalMSEGBTadptFA=totalMSEGBTadptFA./(rBlock*cBlock);

totalPEGGBTadptF=totalPEGGBTadptF./(rBlock*cBlock);
totalMSEGBTadptF=totalMSEGBTadptF./(rBlock*cBlock);

totalPEGTMFA=totalPEGTMFA./(rBlock*cBlock);
totalMSETMFA=totalMSETMFA./(rBlock*cBlock);

totalPEGTMFB=totalPEGTMFB./(rBlock*cBlock);
totalMSETMFB=totalMSETMFB./(rBlock*cBlock);



totalPEGBTONLLap=totalPEGBTONLLap./(rBlock*cBlock);
totalMSEGBTONLLap=totalMSEGBTONLLap./(rBlock*cBlock);

totalPEDCT(1)=0;
 totalMSEDCT(1)=1;
totalPEKLT(1)=0;
 totalMSEKLT(1)=1;
totalPEG(1)=0;
 totalMSEG(1)=1;
 
 totalPEG_ORT(1)=0;
 totalMSEG_ORT(1)=1;
 
 totalPEG_2D(1)=0;
 totalMSEG_2D(1)=1;
 
 totalPEG_NLM(1)=0;
 totalMSEG_NLM(1)=1;
 
 totalPEG_NLM_2D(1)=0;
 totalMSEG_NLM_2D(1)=1;
 
 totalPEG_TMB(1)=0;
 totalMSEG_TMB(1)=1;
 
 totalPEG_TMB_2D(1)=0;
 totalMSEG_TMB_2D(1)=1;
 
 totalPEG_TMBNLM(1)=0;
 totalMSEG_TMBNLM(1)=1;
 
  totalPEG_TMBNLM_2D(1)=0;
 totalMSEG_TMBNLM_2D(1)=1;
 
  
  totalPEG_TMBNLM_2D_old(1)=0;
 totalMSEG_TMBNLM_2D_old(1)=1;
 
 totalPEG_PI(1)=0;
 totalMSEG_PI(1)=1;
 
 totalPEG_PI_2D(1)=0;
 totalMSEG_PI_2D(1)=1;
 
totalPEGr(1)=0;
 totalMSEGr(1)=1;
 
 totalPEGrNN(1)=0;
 totalMSEGrNN(1)=1;
 
 totalPEGrNN_pred(1)=0;
 totalMSEGrNN_pred(1)=1;
 
 totalPEGr_2D(1)=0;
 totalMSEGr_2D(1)=1;
 
  totalPEGr_2D_old(1)=0;
 totalMSEGr_2D_old(1)=1;
 %{
 totalPEGE(1)=0;
 totalMSEGE(1)=1;
totalPEGrA(1)=0;
 totalMSEGrA(1)=1;
 %}
totalPEASI(1)=0;
 totalMSEASI(1)=1;
 totalPEGGBTadptFA(1)=0;
 totalMSEGBTadptFA(1)=1;
 
 totalPEGGBTadptF(1)=0;
 totalMSEGBTadptF(1)=1;
 
 totalPEGTMFA(1)=0;
 totalMSETMFA(1)=1;
 
 totalPEGTMFB(1)=0;
 totalMSETMFB(1)=1;
 
 totalPEGBTONLLap(1)=0;
 totalMSEGBTONLLap(1)=1;

% for coeff=1:1:rb*cb
%     if totalValuesPE(coeff)>0
%         totalPEASI(coeff)=totalPEASI(coeff)/totalValuesPE(coeff);
%     end
%     if totalValuesMSE(coeff)>0
%         totalMSEASI(coeff)=totalMSEASI(coeff)/totalValuesMSE(coeff);
%     end 
% end
% 
% for coeff=2:1:rb*cb
%    
%     if totalPEASI(coeff)==0
%         totalPEASI(coeff)=NaN;
%     end
%     
%     if totalMSEASI(coeff)==0
%         totalMSEASI(coeff)=NaN;
%     end
%     
% end

figure;
hold on
plot((totalPEDCT),'b')%DCT
plot((totalPEKLT),'y')%KLT
plot((totalPEASI),'g')%ASI/DST

%plot((totalPEG),'r') %predicted residual by template matching in residual domain with optimization
plot((totalPEG_ORT),'--r')
%plot((totalPEG_2D),'-.r') %predicted residual by template matching type in residual domain with optimization 2D self loop

%plot((totalPEG_NLM),'c') %predicted residual by template matching in residual domain with weighted average/nlm 
%plot((totalPEG_NLM_2D),'c') %predicted residual by template matching in residual domain with weighted average/nlm 2D

%plot((totalPEG_TMB),'b')%predicted residual by template matching in pixel domain with optimization
%plot((totalPEG_TMB_2D),'-.b')%predicted residual by template matching in pixel domain with optimization 2D self loop

plot((totalPEG_TMBNLM),'-.g')%%predicted residual by template matching in pixel domain with weighted average/nlm 
plot((totalPEG_TMBNLM_2D),'-.b')%%predicted residual by template matching in pixel domain with weighted average/nlm 2D self loop ----now cyan
plot((totalPEG_TMBNLM_2D_old),'c')
plot((totalPEG_PI),'m') %predicted residual by prediction inacc
plot((totalPEG_PI_2D),'-.m') %predicted residual by prediction inacc in 2D

plot((totalPEGBTONLLap),'--c')
plot((totalPEGr),'k') %black actual residual
plot((totalPEGr_2D),'-.k') 
plot((totalPEGr_2D_old),'r') %black actual residual 2D self loop
plot((totalPEGrNN),'-.r') %black actual residual NN

plot((totalPEGrNN_pred),'-.c')
%plot((totalPEGGBTadptFA),'--m')%predicted residual by prediction inacc separable
%plot((totalPEGGBTadptF),'--k')  %black actual residual separable
%plot((totalPEGTMFA),'--r') %predicted residual by template matching type A separable
%plot((totalPEGTMFB),'--y') %predicted residual by template matching type B separable
%{
plot((totalPEGE),'y')
plot((totalPEGrA),'m')
%}
%%plot((totalPEDST),'m')


axis('tight');
%pi=print_img(PE,600,400);

figure;
hold on 
plot(totalMSEDCT,'b')
plot(totalMSEASI,'g')
plot(totalMSEKLT,'y')


%plot(totalMSEG,'r')
plot((totalMSEG_ORT),'--r')
%plot(totalMSEG_2D,'-.r')

%plot(totalMSEG_NLM,'c')
%plot(totalMSEG_NLM_2D,'-.c')

%plot(totalMSEG_TMB,'b')
%plot(totalMSEG_TMB_2D,'-.b')

plot(totalMSEG_TMBNLM,'-.g')
plot(totalMSEG_TMBNLM_2D,'-.b')
plot(totalMSEG_TMBNLM_2D_old,'c')

%plot(totalMSEG_PI,'m')
%plot(totalMSEG_PI_2D,'-.m')
plot(totalMSEGr,'k') %black

plot(totalMSEGrNN,'-.r') %black

plot(totalMSEGrNN_pred,'-.c') %black
plot(totalMSEGBTONLLap,'--c') %black
plot(totalMSEGr_2D,'-.k')% blue 
plot(totalMSEGr_2D_old,'r')
%plot(totalMSEGBTadptFA,'m')%predicted residual by prediction inacc GBST
%plot(totalMSEGBTadptF,'k')  %black actual residual separable
%plot(totalMSETMFA,'r') %separable
%plot(totalMSETMFB,'y') %separable
%{
plot(totalMSEGE,'--y')
plot(totalMSEGrA,'--m')
%}

axis('tight');

toc