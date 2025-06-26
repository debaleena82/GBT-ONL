%srcFiles = dir('Adj_dataset_self\Vert\*.mat');
%srcFiles = dir('DCC2019finalcode\GBT-NNS\Normalised\Vert\*.mat');
%srcFiles = dir('DCC2019finalcode\GBT-NNS\Unnormalised\Horizontal\*.mat');
%srcFiles = dir('DCC2019finalcode\GBT-NNS\Unnormalised\Horizontal\Test\*.mat');
srcFiles = dir('DCC2019finalcode\GBT-ONL\dataset\*.mat');
%srcFiles = dir('DCC2019finalcode\Results_DCC2021\training dataset\Horizontal\*.mat');
%R_Mat_New = zeros(size(srcFiles,1), 4096);

for i11 = 1:size(srcFiles,1)
  %  FileName = strcat('Adj_dataset_self\Vert\', srcFiles(i11).name);
  %FileName = strcat('DCC2019finalcode\GBT-NNS\Normalised\Vert\', srcFiles(i11).name);
   FileName = strcat('D:\MATLAB\DCC2019finalcode\GBT-ONL\dataset\', srcFiles(i11).name);
   % FileName = strcat('DCC2019finalcode\Results_DCC2021\training dataset\Horizontal\', srcFiles(i11).name);
    load(FileName)
 %   size(R_Mat_New)
   % R_Mat_New(i11,:) = res_Cov_Final;
%%%%    R_Mat_New = res_Cov_Final;
%%%%    eval(['COV' num2str(i11) ' =R_Mat_New']);
  %  R_Mat_Adj = res_Adj_Final;
  %  eval(['ADJ' num2str(i11) ' =R_Mat_Adj']);
  %  R_Mat_WL = res_WL_Final;
  %  eval(['WL' num2str(i11) ' =R_Mat_WL']);
%    R_Mat_VLadj = res_VLadj_Final;
%    eval(['VLadj' num2str(i11) ' =R_Mat_VLadj']);
%%%    R_Mat_VLallcov = res_VLallcov_Final;
%%%    eval(['VLallcov' num2str(i11) ' =R_Mat_VLallcov']);
    Avg_Ref_New_Final_temp1 = Avg_Ref_New_Final;
    eval(['Avg_Ref_New_Final_Mat' num2str(i11) ' =Avg_Ref_New_Final_temp1']);
    
   Lrefavg_Final_temp1 = Lrefavg_Final;
    eval(['Lrefavg_Final_Mat' num2str(i11) ' =Lrefavg_Final_temp1']);
 %   R_Mat_VLall = res_VLall_Final;
  %%%  R_Mat_VLall_self = res_VLall_self_Final;
 %   eval(['VLall' num2str(i11) ' =R_Mat_VLall']);
 %%%   eval(['VLall_self' num2str(i11) ' =R_Mat_VLall_self']);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Block_Ref_3D_Comb_temp1 = Block_Ref_3D_Comb;
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% eval(['Block_Ref_3D_Comb_Mat' num2str(i11) ' =Block_Ref_3D_Comb_temp1']);
  %%%%%  R_mode_mat = mode_matrix;
  %%%%%  eval(['mode_mat' num2str(i11) ' =R_mode_mat']);
end
%%%%CC=[COV1];
%CC=[COV1;COV2;COV3;COV4;COV5;COV6;COV7;COV8;COV9;COV10;COV11];
%CC=[COV1;COV2;COV3;COV4;COV5;COV6;COV7;COV8;COV9];
%AD=[ADJ1;ADJ2;ADJ3];
%WW=[WL1;WL2;WL3;WL4;WL5;WL6;WL7;WL8;WL9;WL10;WL11];
%VAD=[VLadj1;VLadj2;VLadj3;VLadj4;VLadj5;VLadj6;VLadj7;VLadj8;VLadj9];
%VAD=[VLadj1];
%VLALLCOV=[VLallcov1];
%VLALLCOV_PRED_COMBO_FRAME=[VLallcov_pred1;VLallcov_pred2;VLallcov_pred3;VLallcov_pred4;VLallcov_pred5;VLallcov_pred6];
%%%%Block_Adj3dgt_3D_Mat_Frame=[Block_Adj3dgt_3D_Mat1;Block_Adj3dgt_3D_Mat2,Block_Adj3dgt_3D_Mat3;Block_Adj3dgt_3D_Mat4;Block_Adj3dgt_3D_Mat5;Block_Adj3dgt_3D_Mat6,Block_Adj3dgt_3D_Mat7;Block_Adj3dgt_3D_Mat8;Block_Adj3dgt_3D_Mat9;Block_Adj3dgt_3D_Mat10,Block_Adj3dgt_3D_Mat11;Block_Adj3dgt_3D_Mat12;Block_Adj3dgt_3D_Mat13;Block_Adj3dgt_3D_Mat14,Block_Adj3dgt_3D_Mat15];

%%%%%%%%%%%%%%%%Block_Adj3dgt_3D_Mat_Frame=cat(3,Block_Adj3dgt_3D_Mat1,Block_Adj3dgt_3D_Mat2,Block_Adj3dgt_3D_Mat3,Block_Adj3dgt_3D_Mat4,Block_Adj3dgt_3D_Mat5,Block_Adj3dgt_3D_Mat6,Block_Adj3dgt_3D_Mat7,Block_Adj3dgt_3D_Mat8,Block_Adj3dgt_3D_Mat9,Block_Adj3dgt_3D_Mat10,Block_Adj3dgt_3D_Mat11,Block_Adj3dgt_3D_Mat12,Block_Adj3dgt_3D_Mat13,Block_Adj3dgt_3D_Mat14,Block_Adj3dgt_3D_Mat15);
Avg_Ref_New_Final_Mat_Frame=cat(1,Avg_Ref_New_Final_Mat1,Avg_Ref_New_Final_Mat2,Avg_Ref_New_Final_Mat3,Avg_Ref_New_Final_Mat4,Avg_Ref_New_Final_Mat5,Avg_Ref_New_Final_Mat6,Avg_Ref_New_Final_Mat7,Avg_Ref_New_Final_Mat8,Avg_Ref_New_Final_Mat9,Avg_Ref_New_Final_Mat10,Avg_Ref_New_Final_Mat11,Avg_Ref_New_Final_Mat12,Avg_Ref_New_Final_Mat13,Avg_Ref_New_Final_Mat14,Avg_Ref_New_Final_Mat15,Avg_Ref_New_Final_Mat16,Avg_Ref_New_Final_Mat17);
%Block_Adj3dgt_3D_Mat_Frame=cat(3,Block_Adj3dgt_3D_Mat1,Block_Adj3dgt_3D_Mat2);
Lrefavg_Final_Mat_Frame= cat(1,Lrefavg_Final_Mat1,Lrefavg_Final_Mat2,Lrefavg_Final_Mat3,Lrefavg_Final_Mat4,Lrefavg_Final_Mat5,Lrefavg_Final_Mat6,Lrefavg_Final_Mat7,Lrefavg_Final_Mat8,Lrefavg_Final_Mat9,Lrefavg_Final_Mat10,Lrefavg_Final_Mat11,Lrefavg_Final_Mat12,Lrefavg_Final_Mat13,Lrefavg_Final_Mat14,Lrefavg_Final_Mat15,Lrefavg_Final_Mat16,Lrefavg_Final_Mat17);
%VALL=[VLall1;VLall2;VLall3;VLall4;VLall5;VLall6;VLall7;VLall8;VLall9];
%VALL=[VLall1];
%VALL_SELF=[VLall_self1];
%VALL_SELF=[VLall_self1];
%VALL_SELF_PRED_COMBO_FRAME=[VLall_self_pred1;VLall_self_pred2;VLall_self_pred3;VLall_self_pred4;VLall_self_pred5;VLall_self_pred6];

%%%%%%%%%%%%%%%%%%Block_Ref_3D_Comb_Mat_Frame=cat(3,Block_Ref_3D_Comb_Mat1,Block_Ref_3D_Comb_Mat2,Block_Ref_3D_Comb_Mat3,Block_Ref_3D_Comb_Mat4,Block_Ref_3D_Comb_Mat5,Block_Ref_3D_Comb_Mat6,Block_Ref_3D_Comb_Mat7,Block_Ref_3D_Comb_Mat8,Block_Ref_3D_Comb_Mat9,Block_Ref_3D_Comb_Mat10,Block_Ref_3D_Comb_Mat11,Block_Ref_3D_Comb_Mat12,Block_Ref_3D_Comb_Mat13,Block_Ref_3D_Comb_Mat14,Block_Ref_3D_Comb_Mat15);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Block_Ref_3D_Comb_Mat_Frame=cat(3,Block_Ref_3D_Comb_Mat1,Block_Ref_3D_Comb_Mat2,Block_Ref_3D_Comb_Mat3,Block_Ref_3D_Comb_Mat4,Block_Ref_3D_Comb_Mat5,Block_Ref_3D_Comb_Mat6,Block_Ref_3D_Comb_Mat7,Block_Ref_3D_Comb_Mat8,Block_Ref_3D_Comb_Mat9,Block_Ref_3D_Comb_Mat10);
%Block_Ref_3D_Comb_Mat_Frame=cat(3,Block_Ref_3D_Comb_Mat1,Block_Ref_3D_Comb_Mat2);

%%%%%Block_Ref_3D_Comb_Mat_Frame=[Block_Ref_3D_Comb_Mat1;Block_Ref_3D_Comb_Mat2;Block_Ref_3D_Comb_Mat3;Block_Ref_3D_Comb_Mat4;Block_Ref_3D_Comb_Mat5;Block_Ref_3D_Comb_Mat6;Block_Ref_3D_Comb_Mat7;Block_Ref_3D_Comb_Mat8;Block_Ref_3D_Comb_Mat9;Block_Ref_3D_Comb_Mat10;Block_Ref_3D_Comb_Mat11;Block_Ref_3D_Comb_Mat12;Block_Ref_3D_Comb_Mat13;Block_Ref_3D_Comb_Mat14;Block_Ref_3D_Comb_Mat15];
%WW=[WL1];
%CC=[COV1];
%AD=[ADJ1];
%MODE_MAT_FRAME=[mode_mat1;mode_mat2;mode_mat3;mode_mat4;mode_mat5;mode_mat6];
%%%%%%%%%%%%%%%%%%MODE_MAT_FRAME=[mode_mat1];
%save('Data_Matrix_New.mat', 'R_Mat_New');
%save('Data_Matrix_New.mat', 'CC','AD');
%%%%save('Data_Matrix_VAD_Vert_self.mat', 'CC','VAD','VALL');

%%%save('Data_Matrix_GBTNNS.mat', 'VLALLCOV_PRED_COMBO_FRAME','VALL_SELF_PRED_COMBO_FRAME','MODE_MAT_FRAME');
%save('Data_Matrix_GBTNNS_Horiz_UN_Test.mat', 'Block_Adj3dgt_3D_Mat_Frame','Block_Ref_3D_Comb_Mat_Frame');
save('Data_Matrix_GBTONL.mat', 'Avg_Ref_New_Final_Mat_Frame','Lrefavg_Final_Mat_Frame');
%save('Data_Matrix_peopleonstreet_ClassA_bestmode.mat', 'VLALLCOV','VALL_SELF');
clear
load Data_Matrix_GBTONL
%load Data_Matrix_VALL_Horz256_self