
% WEIGHT
grot=vars(Kp,[nets_cellfind(varsHeader,'Weight (0.0)'), ...
    nets_cellfind(varsHeader,'Weight (1.0)'),...
    nets_cellfind(varsHeader,'Weight (pre-imaging) (2.0)')]);
WEIGHT = [];
for s = 1:length(Kp)
    k = max(find(isnan(grot(s,:)) == 0));
    if isempty(k)
        WEIGHT(s,1) = NaN;
        WEIGHT(s,2) = NaN;
    else
        WEIGHT(s,1) = grot(s,k);
        WEIGHT(s,2) = k-1;  % visit 0, 1 or 2
    end
end
 
% WAIST
grot=vars(Kp,[nets_cellfind(varsHeader,'Waist circumference (0.0)'), ...
    nets_cellfind(varsHeader,'Waist circumference (1.0)'),...
    nets_cellfind(varsHeader,'Waist circumference (2.0)')]);
WAIST = [];
for s = 1:length(Kp)
    k = max(find(isnan(grot(s,:)) == 0));
    if isempty(k)
        WAIST(s,1) = NaN;
        WAIST(s,2) = NaN;
    else
        WAIST(s,1) = grot(s,k);
        WAIST(s,2) = k-1; % visit 0, 1 or 2
    end
end
 
% HIP
grot=vars(Kp,[nets_cellfind(varsHeader,'Hip circumference (0.0)'), ...
    nets_cellfind(varsHeader,'Hip circumference (1.0)'),...
    nets_cellfind(varsHeader,'Hip circumference (2.0)')]);
HIP = [];
for s = 1:length(Kp)
    k = max(find(isnan(grot(s,:)) == 0));
    if isempty(k)
        HIP(s,1) = NaN;
        HIP(s,2) = NaN;
    else
        HIP(s,1) = grot(s,k);
        HIP(s,2) = k-1; % visit 0, 1 or 2
    end
end
 
% BMI
grot=vars(Kp,[nets_cellfind(varsHeader,'Body mass index (BMI) (0.0)'), ...
    nets_cellfind(varsHeader,'Body mass index (BMI) (1.0)'),...
    nets_cellfind(varsHeader,'Body mass index (BMI) (2.0)')]);
BMI = []; BMIvisit = [];
for s = 1:length(Kp)
    k = max(find(isnan(grot(s,:)) == 0));
    if isempty(k)
        BMI(s,1) = NaN;
        BMI(s,2) = NaN;
    else
        BMI(s,1) = grot(s,k);
        BMI(s,2) = k-1; % visit 0, 1 or 2
    end
end
 
% BPdia
grot=vars(Kp,[nets_cellfind(varsHeader,'Diastolic blood pressure, automated reading (0',-1), ...
    nets_cellfind(varsHeader,'Diastolic blood pressure, automated reading (1',-1),...
    nets_cellfind(varsHeader,'Diastolic blood pressure, automated reading (2',-1)]);
visit = [0,0,1,1,2,2];
BPdia = [];
for s = 1:length(Kp)
    k = max(find(isnan(grot(s,:)) == 0));
    if isempty(k)
        BPdia(s,1) = NaN;
        BPdia(s,2) = NaN;
    else
        BPdia(s,1) = grot(s,k);
        BPdia(s,2) = visit(k);  % visit 0, 1 or 2
    end
end
 
% BPsys
grot=vars(Kp,[nets_cellfind(varsHeader,'Systolic blood pressure, automated reading (0',-1), ...
    nets_cellfind(varsHeader,'Systolic blood pressure, automated reading (1',-1),...
    nets_cellfind(varsHeader, 'Systolic blood pressure, automated reading (2',-1)]);
visit = [0,0,1,1,2,2];
BPsys = [];
for s = 1:length(Kp)
    k = max(find(isnan(grot(s,:)) == 0));
    if isempty(k)
        BPsys(s,1) = NaN;
        BPsys(s,2) = NaN;
    else
        BPsys(s,1) = grot(s,k);
        BPsys(s,2) = visit(k);  % visit 0, 1 or 2
    end
end
 
% SMOKING
grot=vars(Kp,[nets_cellfind(varsHeader,'Past tobacco smoking (0.0)'), ...
    nets_cellfind(varsHeader,'Past tobacco smoking (1.0)'),...
    nets_cellfind(varsHeader,'Past tobacco smoking (2.0)')]);
SMOKING = [];
for s = 1:length(Kp)
    k = max(find(isnan(grot(s,:)) == 0));
    if isempty(k)
        SMOKING(s,1) = NaN;
        SMOKING(s,2) = NaN;
    else
        SMOKING(s,1) = grot(s,k);
        SMOKING(s,2) = k-1; % visit 0, 1 or 2
    end
end
 
% ALCOHOL
grot=vars(Kp,[nets_cellfind(varsHeader,'Alcohol intake frequency. (0.0)'), ...
    nets_cellfind(varsHeader,'Alcohol intake frequency. (1.0)'),...
    nets_cellfind(varsHeader,'Alcohol intake frequency. (2.0)')]);
ALCOHOL = [];
for s = 1:length(Kp)
    k = max(find(isnan(grot(s,:)) == 0));
    if isempty(k)
        ALCOHOL(s,1) = NaN;
        ALCOHOL(s,2) = NaN;
    else
        ALCOHOL(s,1) = grot(s,k);
        ALCOHOL(s,2) = k-1; % visit 0, 1 or 2
    end
end
 
% DIABETES
grot1=[nets_cellfind(varsHeader,'Diabetes diagnosed by doctor (0.0)') ...
      nets_cellfind(varsHeader,'Diabetes diagnosed by doctor (1.0)') ...
      nets_cellfind(varsHeader,'Diabetes diagnosed by doctor (2.0)') ...
      nets_cellfind(varsHeader,'Non-cancer illness code, self-reported (1220 - diabetes)') ...
      nets_cellfind(varsHeader,'Non-cancer illness code, self-reported (1222 - type 1 diabetes)') ...
      nets_cellfind(varsHeader,'Non-cancer illness code, self-reported (1223 - type 2 diabetes)')];
grot2=[nets_cellfind(varsHeader,'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones (0.0)')...
      nets_cellfind(varsHeader,'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones (0.1)')...
      nets_cellfind(varsHeader,'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones (0.2)')...
      nets_cellfind(varsHeader,'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones (1.0)')...
      nets_cellfind(varsHeader,'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones (1.1)')...
      nets_cellfind(varsHeader,'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones (2.0)')...
      nets_cellfind(varsHeader,'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones (2.1)')...
      nets_cellfind(varsHeader,'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones (2.2)')...
      nets_cellfind(varsHeader,'Medication for cholesterol, blood pressure or diabetes (0.0)')...
      nets_cellfind(varsHeader,'Medication for cholesterol, blood pressure or diabetes (0.1)')...
      nets_cellfind(varsHeader,'Medication for cholesterol, blood pressure or diabetes (1.0)')...
      nets_cellfind(varsHeader,'Medication for cholesterol, blood pressure or diabetes (1.1)')...
      nets_cellfind(varsHeader,'Medication for cholesterol, blood pressure or diabetes (2.0)')...
      nets_cellfind(varsHeader,'Medication for cholesterol, blood pressure or diabetes (2.1)')];
 
DIABETES1=max(vars(Kp,grot1),[],2); % diabetes from doctor and self
DIABETES2=max(vars(Kp,grot2) == 3,[],2); % diabetes from insulin medication
DIABETES=max([DIABETES1, DIABETES2],[],2);  % take union
% [ sum(DIABETES(CVCV==0)) sum(DIABETES(CVCV==1)) ]


grot=vars(:,[nets_cellfind(varsVARS,'6153-0.0') nets_cellfind(varsVARS,'6153-0.1') nets_cellfind(varsVARS,'6153-0.2') nets_cellfind(varsVARS,'6153-1.0') ...
  nets_cellfind(varsVARS,'6153-1.1') nets_cellfind(varsVARS,'6153-2.0') nets_cellfind(varsVARS,'6153-2.1') nets_cellfind(varsVARS,'6153-2.2') ...
  nets_cellfind(varsVARS,'6177-0.0') nets_cellfind(varsVARS,'6177-0.1') nets_cellfind(varsVARS,'6177-1.0') nets_cellfind(varsVARS,'6177-1.1') ...
  nets_cellfind(varsVARS,'6177-2.0') nets_cellfind(varsVARS,'6177-2.1')]);
grot1=zeros(size(grot)); grot1(grot==1)=1; grot1=max(grot1,[],2); %sum(grot1) % cholesterol meds
grot2=zeros(size(grot)); grot2(grot==2)=1; grot2=max(grot2,[],2); %sum(grot2)
grot3=zeros(size(grot)); grot3(grot==3)=1; grot3=max(grot3,[],2); %sum(grot3)
grot4=zeros(size(grot)); grot4(grot==4)=1; grot4=max(grot4,[],2); %sum(grot4)
grot5=zeros(size(grot)); grot5(grot==5)=1; grot5=max(grot5,[],2); %sum(grot5)
CHOL=grot1(Kp);

