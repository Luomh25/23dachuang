
% Dependencies: FSLNets, dscatter, nancorr

% Note that a lot of the opening code is specific to your own version
% (and organisation) of variables directly available from UK Biobank,
% and as such will need amending according to your organisation of
% data.

ColControl=[58 175 186]/256; ColCase=[254 168 119]/256;

%%% main options
DoTime=0;           % longitudinal or cross-sectional options:  0=time2-time1 1=time1 2=time2
DoModulate=1;       % age-modulate case-control regressor (1), or not (0) ?
DoHypPriors=0;      % restrict IDP list to a priori set of hypothesised IDPs
Nperm=10000;        % number of permutations
DoTime0IDPsSubs=1;  % select IDPs and subjects as if for DoTime=0

%%% load new IDPs for ALL subjects ALL timepoints
% subject_IDs is the full list of UKB subject IDs in your workspace;
% this is one ID per *session* where the actual unique ID
% (subject_IDs_unique) is prepended by "2" or "3" according to visit
% number. Each of these new IDP data files has subject_ID as the first
% column.
OB=nets_load_match('UKB_OB_HT_PG_IDPs_v7.txt',subject_IDs);
QSM=nets_load_match('QSM_v5.txt',subject_IDs);
ASL=nets_load_match('ASL_IDPs.txt',subject_IDs);
FSsubseg=nets_load_match('FSsubseg4.txt',subject_IDs);
MDperc90=nets_load_match('MDperc90.txt',subject_IDs);
grot=[IDPs OB QSM ASL FSsubseg MDperc90]; Z1mean=nanmedian(grot); clear OB QSM ASL FSsubseg MDperc90;
grot=grot-nanmedian(grot); grotmedabs=nanmedian(abs(grot)); grotmedabs(grotmedabs<eps)=nanstd(grot(:,grotmedabs<eps))/1.48; Z1std=grotmedabs;
grot=grot./grotmedabs; grot(abs(grot)>8)=NaN;  % nanstd part is for pathological distributions
IDPsCOVID=grot;

%%% expand IDP_names (the list of IDP name strings) to include new custom IDPs
NEWIDPS_info
for i=1:(size(IDPsCOVID,2)-length(IDP_categories));
  IDP_categories(length(IDP_categories)+1)=length(IDP_category_names)+1;
end
IDP_category_names{length(IDP_category_names)+1}='New IDPs';

%%% Hypothesis-led list of a-priori-defined IDPs to evaluate
DoHypList; TheHypList=[];
for i=1:length(HypList)
  grot=nets_cellfind(IDP_names,HypList{i});
  TheHypList=[TheHypList grot];
end

%%% load up diagnosis date (need to download this info from UKB)
grot=load('coviddiagdate_2021-08-03.mat'); 
[~,grotI1,grotJ1]=intersect(subject_IDs_unique,grot.eids(:,1));
DiagDate=zeros(size(subject_IDs_unique,1),1)/0; DiagDate(grotI1,:)=grot.coviddiagdate(grotJ1); DiagDate(DiagDate==0)=nan;
DiagScanInterval=scan_date2-DiagDate;

%%% load covid outcomes for imaged subjects, info supplied by UKB variable 41000
CV=nets_load_match('var41000.txt',subject_IDs_unique);
CV=mod(CV,10); CV(CV==9)=-1;           % 9=>-1 means don't have full info yet
CV(isnan(CV) & scan_date2<2020.5)=-2;  % pre-pandemic scanning (keep post-covid scans not being used by this DoBatch choice as nan)

%%% results from health records of hospitalisation
Hospitalized = [%your data access application-specific list of subject IDs%];
[~,grot]=intersect(subject_IDs_unique,Hospitalized); Hospitalized=subject_IDs_unique*0; Hospitalized(grot)=1;

%%% separate out two timepoints for IDPs and confounds
[~,grotI1,grotJ1]=intersect(subject_IDs_unique+2e7,subject_IDs);  [~,grotI2,grotJ2]=intersect(subject_IDs_unique+3e7,subject_IDs);
IDPs1COVID=zeros(size(subject_IDs_unique,1),size(IDPsCOVID,2))/0;  IDPs2COVID=IDPs1COVID;
IDPs1COVID(grotI1,:)=IDPsCOVID(grotJ1,:); IDPs2COVID(grotI2,:)=IDPsCOVID(grotJ2,:);
grot=[head_size_scaling conf_TablePos_COG_Table conf_TablePos_COG_Z conf_YTRANS age];
CONF1=zeros(size(subject_IDs_unique,1),size(grot,2))/0; CONF2=CONF1; CONF1(grotI1,:)=grot(grotJ1,:); CONF2(grotI2,:)=grot(grotJ2,:);

%%% how similar are the IDPs between two timepoints?
grot1=zeros(1,size(IDPs1COVID,2))/0; grot2=grot1; grot3=grot1;
for i=1:size(IDPs1COVID,2)
  grot1(i)=nancorr(IDPs1COVID(CV==-2,i),    IDPs2COVID(CV==-2,i));
  grot2(i)=nancorr(IDPs1COVID(CV==0,i),     IDPs2COVID(CV==0,i));
  grot3(i)=nancorr(IDPs1COVID(CV==1,i),     IDPs2COVID(CV==1,i));
end
grotREPROD=(grot2+grot3)/2;  % pooling reproducibility between controls and covid

%%% work out relevant subset of subjects
if DoTime==0 | DoTime0IDPsSubs==1
  X = IDPs2COVID - IDPs1COVID; % temporary differencing of IDPs, just to get info on missing data
else
  if DoTime==1
    X = IDPs1COVID;
  else
    X = IDPs2COVID;
  end
end
Xgood= (sum(~isnan(X),2)>10) & (CV>-1); Kp=find(Xgood==1);  % Kp indexes into the final set of subjects to be used
KpN=length(Kp);

%%% session-specific confs - pool all timepoints, all subjects
X1=IDPs1COVID(Kp,:); X2=IDPs2COVID(Kp,:);
grot=[X1;X2];
  Z2mean=nanmedian(grot); grot=grot-nanmedian(grot); grotmedabs=nanmedian(abs(grot)); 
  grotmedabs(grotmedabs<eps)=nanstd(grot(:,grotmedabs<eps))/1.48; Z2std=grotmedabs; grot=grot./grotmedabs; grot(abs(grot)>8)=NaN;  grot1=grot;
grot=[CONF1(Kp,:);CONF2(Kp,:)];
  grot=grot-nanmedian(grot); grotmedabs=nanmedian(abs(grot));
  grotmedabs(grotmedabs<eps)=nanstd(grot(:,grotmedabs<eps))/1.48; grot=grot./grotmedabs; grot(abs(grot)>15)=NaN; grot(isnan(grot))=0; grot2=grot;
Z3mean=nanmean(grot1);
X=nets_unconfound(grot1,grot2); X1=X(1:KpN,:); X2=X(KpN+1:end,:); X=X2-X1;   % session-level deconfounding and IDP differencing
if DoTime0IDPsSubs==1
  IDPs1COVID(:,sum(~isnan(X))<=50)=nan; IDPs2COVID(:,sum(~isnan(X))<=50)=nan;
end
if DoTime==1, X=IDPs1COVID(Kp,:); end;
if DoTime==2, X=IDPs2COVID(Kp,:); end;

%%% pre-exclude non-relevant IDPs, low reprod IDPs, etc.
grot=ones(1,size(IDP_names,1));
grot(nets_cellfind(IDP_names,'rfMRI connectivity (ICA',-1))=0;
grot(nets_cellfind(IDP_names,'_MO_',-1))=0;
grot(nets_cellfind(IDP_names,'_L1_',-1))=0;grot(nets_cellfind(IDP_names,'_L2_',-1))=0;grot(nets_cellfind(IDP_names,'_L3_',-1))=0;
grot(nets_cellfind(IDP_names,'_threshold=0_',-1))=0;
grot(nets_cellfind(IDP_names,'TUB_',-1))=1;grot(nets_cellfind(IDP_names,'AON_',-1))=1;
grot(nets_cellfind(IDP_names,'PIF_',-1))=1;grot(nets_cellfind(IDP_names,'PIT_',-1))=1;
grot(nets_cellfind(IDP_names,'_max_',-1))=0;grot(nets_cellfind(IDP_names,'_median_intensity_',-1))=0;grot(nets_cellfind(IDP_names,'_90th_',-1))=0;
grot(IDP_categories==2)=0;grot(IDP_categories==7)=0;grot(IDP_categories==14)=0;
if DoHypPriors==1
  grot=grot*0; grot(TheHypList)=1;  % replace listing with TheHypListing
end
REPRODthresh=0.5; grotr=grotREPROD>REPRODthresh;  if DoTime>0 & DoTime0IDPs==0, grotr(isnan(grotREPROD))=1; end;
grot = grot & grotr & sum(~isnan(X))>50; IDP_keep=grot;
X=X(:,IDP_keep); X1=X1(:,IDP_keep); X2=X2(:,IDP_keep); IDP_names_TMP=IDP_names(IDP_keep); grotREPRODtmp=grotREPROD(IDP_keep); IDP_categories_TMP=IDP_categories(IDP_keep);

%%% setup several main variables for the final subset of subjects
AGE1=age1(Kp); AGE1(isnan(AGE1))=nanmean(AGE1);
grot=scan_date2(Kp); grot(isnan(grot))=max(grot); AGE2=grot-birth_date(Kp);
SEX=sex12(Kp);
ETHNICITY=vars(Kp,nets_cellfind(varsVARS,'21000-0.0')); ETHNICITY=(ETHNICITY<1000) | (ETHNICITY>1010);
CVCV=CV(Kp);
HOSP=Hospitalized(Kp);
DSI=DiagScanInterval(Kp);

%%% get the nIDP confound variables (BMI etc)
confound_nIDPs
WEIGHT=WEIGHT(:,1); WAIST=WAIST(:,1); HIP=HIP(:,1); BMI=BMI(:,1); BPdia=BPdia(:,1); BPsys=BPsys(:,1); SMOKING=SMOKING(:,1); ALCOHOL=ALCOHOL(:,1);
j=nets_cellfind(varsHeader,'ownsend',-1); TOWNSEND=vars(Kp,j); TOWNSEND(isnan(TOWNSEND))=nanmean(TOWNSEND);
CONF_bmi_diabetes_diaBP=nets_normalise([BMI(:,1) DIABETES BPdia(:,1)]); CONF_bmi_diabetes_diaBP(isnan(CONF_bmi_diabetes_diaBP))=0;

%%% make "raw" IDPs (deconfounded except for age)
grot=[IDPs1COVID(Kp,:);IDPs2COVID(Kp,:)];
  grot=grot-nanmedian(grot); grotmedabs=nanmedian(abs(grot)); 
  grotmedabs(grotmedabs<eps)=nanstd(grot(:,grotmedabs<eps))/1.48; grot=grot./grotmedabs; grot(abs(grot)>8)=NaN;  grot1=grot;
grot=[CONF1(Kp,:);CONF2(Kp,:)];  grot=grot(:,1:end-1);  % remove age term from deconfounding
  grot=grot-nanmedian(grot); grotmedabs=nanmedian(abs(grot));
  grotmedabs(grotmedabs<eps)=nanstd(grot(:,grotmedabs<eps))/1.48; grot=grot./grotmedabs; grot(abs(grot)>15)=NaN; grot(isnan(grot))=0; grot2=grot;
grot=nets_unconfound(grot1,grot2); grot=grot(:,IDP_keep);
X1raw=grot(1:KpN,:); X2raw=grot(KpN+1:end,:); Xraw=X2raw-X1raw;
%%% X1raw and Z2raw =   ((((Xorig-Z1mean)/Z1std) - Z2meanraw)/Z2stdraw)-Z3meanraw
X1rawORIG = ((((X1raw+Z3mean(IDP_keep)).*Z2std(IDP_keep))+Z2mean(IDP_keep)).*Z1std(IDP_keep))+Z1mean(IDP_keep);
X2rawORIG = ((((X2raw+Z3mean(IDP_keep)).*Z2std(IDP_keep))+Z2mean(IDP_keep)).*Z1std(IDP_keep))+Z1mean(IDP_keep);
X1ORIG =    ((((X1   +Z3mean(IDP_keep))   .*Z2std(IDP_keep))+Z2mean(IDP_keep)).*Z1std(IDP_keep))+Z1mean(IDP_keep);
X2ORIG =    ((((X2   +Z3mean(IDP_keep))   .*Z2std(IDP_keep))+Z2mean(IDP_keep)).*Z1std(IDP_keep))+Z1mean(IDP_keep);
X1meanORIG =    ((((nanmean(X1)+Z3mean(IDP_keep))   .*Z2std(IDP_keep))+Z2mean(IDP_keep)).*Z1std(IDP_keep))+Z1mean(IDP_keep);
betaSCALE =  100 * Z2std(IDP_keep) .* Z1std(IDP_keep) ./ X1meanORIG;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TopIDPs=[];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'PIT_glm_mod=NODDI_OD_threshold=0_warpResolution=2mm')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'TUB_glm_mod=NODDI_ISOVF_threshold=0_warpResolution=2mm')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'PIF_mean_intensity_mod=DTI_MD_threshold=0_warpResolution=2mm')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'PIT_glm_mod=DTI_MD_threshold=0_warpResolution=2mm')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'TUB_glm_mod=DTI_MD_threshold=0_warpResolution=2mm')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'aparc-DKTatlas_lh_thickness_lateralorbitofrontal')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'PIT_glm_mod=NODDI_ISOVF_threshold=0_warpResolution=2mm')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'AON_glm_mod=DTI_MD_threshold=0_warpResolution=2mm')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'wg_lh_intensity-contrast_parahippocampal')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'AON_glm_mod=NODDI_ISOVF_threshold=0_warpResolution=2mm')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'aseg_global_volume-ratio_BrainSegVol-to-eTIV')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'IDP_T1_SIENAX_CSF_normalised_volume')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'aseg_rh_volume_Lateral-Ventricle')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'IDP_dMRI_TBSS_ICVF_Superior_fronto-occipital_fasciculus_R')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'aseg_global_volume_BrainSegNotVentSurf')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'aparc-Desikan_lh_thickness_rostralanteriorcingulate')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'aseg_global_volume_BrainSegNotVent')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'aseg_global_volume_SupraTentorialNotVent')];
TopIDPs=[TopIDPs nets_cellfind(IDP_names_TMP,'IDP_T1_FAST_ROIs_R_cerebellum_crus_II')];
TopIDPs=sort(TopIDPs);  % IDP_names_TMP(TopIDPs)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DMmodulate=10.^(AGE2*0.0524-3.27);  % exact model from: https://link.springer.com/article/10.1007/s10654-020-00698-1
if DoTime==1
  DMmodulate=10.^(AGE1*0.0524-3.27);
end
if DoModulate==0
  DMmodulate=ones(size(DMmodulate));
end
DMmodulate=DMmodulate/mean(DMmodulate);
DM = nets_demean(CVCV).*DMmodulate;

if DoTime==0
  CONF=[AGE2-AGE1 AGE2.^2-AGE1.^2 SEX ETHNICITY DMmodulate];
else
  if DoTime==1, grot=CONF1(Kp,:); else, grot=CONF2(Kp,:); end;
  grot=grot-nanmedian(grot); grotmedabs=nanmedian(abs(grot));
  grotmedabs(grotmedabs<eps)=nanstd(grot(:,grotmedabs<eps))/1.48; grot=grot./grotmedabs; grot(abs(grot)>15)=NaN; grot(isnan(grot))=0;
  if DoTime==1
    CONF=[grot AGE1 AGE1.^2 SEX ETHNICITY DMmodulate];
  else
    CONF=[grot AGE2 AGE2.^2 SEX ETHNICITY DMmodulate];
  end
end

%%% remove outliers in delta-IDP from X1,X2,X
for i=1:size(X2,2)
  grot=nets_unconfound(X2(:,i),[CONF X1(:,i)]);
  grot=grot-nanmedian(grot); grotmedabs=nanmedian(abs(grot)); grotmedabs(grotmedabs<eps)=nanstd(grot(:,grotmedabs<eps))/1.48;
  grot=grot./grotmedabs; grot=abs(grot)>8;
  X(grot,i)=nan; X1(grot,i)=nan; X2(grot,i)=nan;
  X1rawORIG(grot,i)=nan; X2rawORIG(grot,i)=nan; X1ORIG(grot,i)=nan; X2ORIG(grot,i)=nan; 
end;

%%% Model 1: case-control
for i=1:size(X,2)
  grot=~isnan(X(:,i)); grotN(i)=sum(grot);
  if DoTime==0
    [~,~,~,grotPi,~,grotZ(i),grotBi,grotBSEi] = ssglm(nets_demean(X2(grot,i)),nets_demean([DM(grot,:) CONF(grot,:) X1(grot,i)]),[1 zeros(1,size(CONF,2)+1)]');
  else
    [~,~,~,grotPi,~,grotZ(i),grotBi,grotBSEi] = ssglm(nets_demean(X(grot,i)),nets_demean([DM(grot,:) CONF(grot,:)]),[1 zeros(1,size(CONF,2))]');
  end
  grotP(i)=2*grotPi(1); grotBSE(i)=sqrt(grotBSEi)*abs(betaSCALE(i)); grotB(i)=grotBi*betaSCALE(i);
end

%%% Model 2: case-control, excluding Hosp
for i=1:size(X,2)
  grot=HOSP==0;
  Xa=X(grot,:); X2a=X2(grot,:); X1a=X1(grot,:); DMa=nets_demean(CVCV(grot)).*DMmodulate(grot)/mean(DMmodulate(grot)); CONFa=CONF(grot,:);
  grot=~isnan(Xa(:,i)); grotNa(i)=sum(grot);
  if DoTime==0
    [~,~,~,grotPi,~,grotZa(i),grotBi,grotBSEi] = ssglm(nets_demean(X2a(grot,i)),nets_demean([DMa(grot,:) CONFa(grot,:) X1a(grot,i)]),[1 zeros(1,size(CONFa,2)+1)]');
  else
    [~,~,~,grotPi,~,grotZa(i),grotBi,grotBSEi] = ssglm(nets_demean(Xa(grot,i)),nets_demean([DMa(grot,:) CONFa(grot,:)]),[1 zeros(1,size(CONFa,2))]');
  end
  grotPa(i)=2*grotPi(1); grotBSEa(i)=sqrt(grotBSEi)*abs(betaSCALE(i)); grotBa(i)=grotBi*betaSCALE(i);
end

%%% Model 3: hosp-control
for i=1:size(X,2)
  grot=CVCV==0 | HOSP==1; Xb=X(grot,:); X2b=X2(grot,:); X1b=X1(grot,:);
  DMb=nets_demean(CVCV(grot)).*DMmodulate(grot)/mean(DMmodulate(grot)); CONFb=CONF(grot,:);
  grot=~isnan(Xb(:,i)); grotNb(i)=sum(grot);
  if DoTime==0
    [~,~,~,grotPi,~,grotZb(i),grotBi,grotBSEi] = ssglm(nets_demean(X2b(grot,i)),nets_demean([DMb(grot,:) CONFb(grot,:) X1b(grot,i)]),[1 zeros(1,size(CONFb,2)+1)]');
  else
    [~,~,~,grotPi,~,grotZb(i),grotBi,grotBSEi] = ssglm(nets_demean(Xb(grot,i)),nets_demean([DMb(grot,:) CONFb(grot,:)]),[1 zeros(1,size(CONFb,2))]');
  end
  grotPb(i)=2*grotPi(1); grotBSEb(i)=sqrt(grotBSEi)*abs(betaSCALE(i)); grotBb(i)=grotBi*betaSCALE(i);
end

%%% Model 4: hosp-covid
for i=1:size(X,2)
  grot=CVCV==1; Xc=X(grot,:); X2c=X2(grot,:); X1c=X1(grot,:);
  DMc=nets_demean(HOSP(grot)).*DMmodulate(grot)/mean(DMmodulate(grot)); CONFc=[CONF(grot,:) CONF_bmi_diabetes_diaBP(grot,:)];
  grot=~isnan(Xc(:,i)); grotNc(i)=sum(grot);
  if DoTime==0
    [~,~,~,grotPi,~,grotZc(i),grotBi,grotBSEi] = ssglm(nets_demean(X2c(grot,i)),nets_demean([DMc(grot,:) CONFc(grot,:) X1c(grot,i)]),[1 zeros(1,size(CONFc,2)+1)]');
  else
    [~,~,~,grotPi,~,grotZc(i),grotBi,grotBSEi] = ssglm(nets_demean(Xc(grot,i)),nets_demean([DMc(grot,:) CONFc(grot,:)]),[1 zeros(1,size(CONFc,2))]');
  end
  grotPc(i)=2*grotPi(1); grotBSEc(i)=sqrt(grotBSEi)*abs(betaSCALE(i)); grotBc(i)=grotBi*betaSCALE(i);
end

%%% permutation test to get FWE P-values
clear grotPP* grotZZ*;

parfor j=1:Nperm;

  DMr=DM(randperm(size(X,1))); grot=-1;
  for i=1:size(X,2); poop=~isnan(X(:,i));
    if DoTime==0
      [~,~,~,~,~,grotz] = ssglm(nets_demean(X2(poop,i)),nets_demean([DMr(poop,:) CONF(poop,:) X1(poop,i)]),[1 zeros(1,size(CONF,2)+1)]'); grot=max(grot,abs(grotz));
    else
      [~,~,~,~,~,grotz] = ssglm(nets_demean(X(poop,i)),nets_demean([DMr(poop,:) CONF(poop,:)]),[1 zeros(1,size(CONF,2))]'); grot=max(grot,abs(grotz));
    end
  end; grotZZ(j)=grot;

  DMr=DMa(randperm(size(Xa,1))); grot=-1;
  for i=1:size(X,2); poop=~isnan(Xa(:,i));
    if DoTime==0
      [~,~,~,~,~,grotz] = ssglm(nets_demean(X2a(poop,i)),nets_demean([DMr(poop,:) CONFa(poop,:) X1a(poop,i)]),[1 zeros(1,size(CONF,2)+1)]'); grot=max(grot,abs(grotz));
    else
      [~,~,~,~,~,grotz] = ssglm(nets_demean(Xa(poop,i)),nets_demean([DMr(poop,:) CONFa(poop,:)]),[1 zeros(1,size(CONF,2))]'); grot=max(grot,abs(grotz));
    end
  end; grotZZa(j)=grot;

  DMr=DMb(randperm(size(Xb,1))); grot=-1;
  for i=1:size(X,2); poop=~isnan(Xb(:,i));
    if DoTime==0
      [~,~,~,~,~,grotz] = ssglm(nets_demean(X2b(poop,i)),nets_demean([DMr(poop,:) CONFb(poop,:) X1b(poop,i)]),[1 zeros(1,size(CONF,2)+1)]'); grot=max(grot,abs(grotz));
    else
      [~,~,~,~,~,grotz] = ssglm(nets_demean(Xb(poop,i)),nets_demean([DMr(poop,:) CONFb(poop,:)]),[1 zeros(1,size(CONF,2))]'); grot=max(grot,abs(grotz));
    end
  end; grotZZb(j)=grot;

  DMr=DMc(randperm(size(Xc,1))); grot=-1;
  for i=1:size(X,2); poop=~isnan(Xc(:,i));
    if DoTime==0
      [~,~,~,~,~,grotz] = ssglm(nets_demean(X2c(poop,i)),nets_demean([DMr(poop,:) CONFc(poop,:) X1c(poop,i)]),[1 zeros(1,size(CONFc,2)+1)]'); grot=max(grot,abs(grotz));
    else
      [~,~,~,~,~,grotz] = ssglm(nets_demean(Xc(poop,i)),nets_demean([DMr(poop,:) CONFc(poop,:)]),[1 zeros(1,size(CONFc,2))]'); grot=max(grot,abs(grotz));
    end
  end; grotZZc(j)=grot;
end

for i=1:length(grotP); grotPP(i)  = (1+sum(grotZZ   >=abs(grotZ(i))))  / (1+Nperm); end;
for i=1:length(grotP); grotPPa(i) = (1+sum(grotZZa  >=abs(grotZa(i)))) / (1+Nperm); end;
for i=1:length(grotP); grotPPb(i) = (1+sum(grotZZb  >=abs(grotZb(i)))) / (1+Nperm); end;
for i=1:length(grotP); grotPPc(i) = (1+sum(grotZZc  >=abs(grotZc(i)))) / (1+Nperm); end;

