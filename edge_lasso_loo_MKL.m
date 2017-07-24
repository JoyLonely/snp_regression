load('5.mat');
load('VBM_Dcerebellum2.mat');

VBM_2 = jb_normalizeNet(VBM_2,1);
NET = jb_normalizeNet(NET,1);
sigma=10;
weight_net = zeros(4005,1);
weight_vbm = zeros(90,1);
b = [];
dim = [];
ccc = [];
rmsee = [];
snp_num = 27;

for lamda1 = 5:10:85%0.8%:0.2:1
    for lamda2 = 5:200:3805%0.4%:0.2:1
%         for Penalty1 = 0.8%:0.2:1
%             for Penalty2 = 0.2%:0.2:1
                cc = [];
                regr_snp = [];
                regr_pre = [];
                m = [];mm = [];
                msee = [];
for c1 = 0:0.1:1
    c2 = 1-c1;

for i = 1:64
    train_net = NET;
    train_net(i,:) = [];
    test_net = NET(i,:);
    
    train_VBM = VBM_2;
    train_VBM(i,:) = [];
    test_VBM = VBM_2(i,:);
    
    train_snp = SNPDATA(:,snp_num);
    train_snp(i,:) = [];
    test_snp = SNPDATA(i,snp_num);
            
    ground = SCORE(:,5);
    ground(i,:) = [];
    

      %% lasso
%         opts.rsL2 = Penalty1;
%         opts.rFlag=1;
%         opts.mFlag=0;
%         opts.lFlag=0;
%         [aaa,funVal] = nnLeastR(train_net,train_snp,lamda1,opts);%nnLeastR(A, y, λ, opts)
%         b = aaa==0;
%         train_net(:,b) = []; 
%         test_net(:,b) = []; 
%         dim = [dim;length(find(b==0))];
%         weight_net = weight_net+aaa;   
%         
%         opts.rsL2 = Penalty2;
%         [aaa,funVal] = nnLeastR(train_VBM,train_snp,lamda2,opts);
%         b = aaa==0;
%         train_VBM(:,b) = []; 
%         test_VBM(:,b) = []; 
%         dim = [dim;length(find(b==0))];
%         weight_vbm = weight_vbm+aaa;   
       %% RFE特征选择
%         dim = [];
%         r = SVMRFE(ground, train_net);
%         dim = r(4096-lamda:4095);
%         train_net=train_net(:,dim);
%         test_net=test_net(:,dim);
%         for k=1:length(dim)  
%             weight(dim(k)) = weight(dim(k))+1;         %统计每一维被选中的次数
%         end

      %% t-test特征选取
%       b = ground==1;
%       dim = [];
%       for f = 1:4005
%           cl_1 = train_net(b,f);
%           cl_2 = train_net(~b,f);
%           if ttest2(cl_1,cl_2) %如果T-TEST返回1代表该特征有判别性，保留该维特征
%               dim = [dim;1];
%           else
%               dim = [dim;0];
%           end
%       end
%       train_net(:,~dim) = []; 
%       test_net(:,~dim) = []; 
%       d = length(find(dim==1));  %记录降维之后的维数
%       weight_net = weight_net+dim;         %统计每一维被选中的次数
%       
%       b = ground==1;
%       dim = [];
%       for f = 1:90
%           cl_1 = train_VBM(b,f);
%           cl_2 = train_VBM(~b,f);
%           if ttest2(cl_1,cl_2) %如果T-TEST返回1代表该特征有判别性，保留该维特征
%               dim = [dim;1];
%           else
%               dim = [dim;0];
%           end
%       end
%       train_VBM(:,~dim) = []; 
%       test_VBM(:,~dim) = []; 
%       d = length(find(dim==1));  %记录降维之后的维数
%       weight_vbm = weight_vbm+dim;         %统计每一维被选中的次数
 %% PCA降维
       options=[];
       options.ReducedDim=lamda1; %降到lamda1维
	   [eigvector,eigvalue] = PCA(train_VBM,options);
       test_VBM = test_VBM*eigvector;
       train_VBM = train_VBM*eigvector;
       options.ReducedDim=lamda2; %降到lamda2维
	   [eigvector,eigvalue] = PCA(train_net,options);
       test_net = test_net*eigvector;
       train_net = train_net*eigvector;

    train_net=train_net./repmat(sqrt(sum(train_net.^2,2)),[1 size(train_net,2)]);
    train_VBM=train_VBM./repmat(sqrt(sum(train_VBM.^2,2)),[1 size(train_VBM,2)]); 
    nK=calckernel('linear',sigma,train_net);
    nKt=calckernel('linear',sigma,train_net,test_net);
    vK=calckernel('linear',sigma,train_VBM);
    vKt=calckernel('linear',sigma,train_VBM,test_VBM);
    K=c1*nK+c2*vK;
    Kt=c1*nKt+c2*vKt;
    [bestmse,bestc,bestg] = SVMcgForRegress(train_snp,K,0,10,-10,0,5);
    cmd = ['-s 4 ',' -c ', num2str(bestc), ' -g ', num2str(bestg)];
%     cmd = ['-s 4 '];
    mod = svmtrain(train_snp,K,cmd);
    [py,mse] = svmpredict(test_snp,Kt,mod);
    mm = [mm;mse(2)];
    regr_snp = [regr_snp;test_snp];
    regr_pre = [regr_pre;py];
     
end
% rmse = sqrt(sum((regr_pre-regr_snp).^2)/length(regr_pre));
% m = [m;rmse];
mean_m = mean(mm);
coef = corrcoef(regr_pre,regr_snp);
corr = abs(coef(1,2));
msee = [msee;mean_m];
cc = [cc;corr];
end
% cc = [cc;lamda1;lamda2;Penalty1;Penalty2];
% cc = [cc;c1;c2];%无参
cc = [cc;lamda1;lamda2];
ccc = [ccc,cc];
rmsee = [rmsee,msee];
            end
        end
%     end
% end
