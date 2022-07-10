clear;
close all;
% newData = importdata('I:\posture_tremor\postural_action_video\0921崔_术前一天\200921094020.txt', '\t', 2);
newData = readtable('I:\posture_tremor\postural_action_video\1028夏_术后一天\201028104723.txt');
rawdata = table2array(newData(:,4:12));
% 补值后的代码
% newData = importdata('H:\rest_tremor\data\1202刘_筛查\201019152517.txt', '\t');


% 使用肖维勒方法（等置信概率）剔除异常值
[m,n] = size(rawdata);
% w = 1 + 0.4*log(m);    % 肖维勒系数（近似计算公式）
% data = size(:,9);
% for i = 1:n
%     left = 1;
% %     data(:,i) = [];
%     for j = left:left+window
%         if left+window < m
%            d = rawdata(left:left+window,i);    
%            data(left:left+window,i) = filloutliers(d,'linear');
%            left = left+window+1;
%         end
%     end
% end

for i = 1:n
    d = rawdata(:,i);
    d = fillmissing(d,'movmedian',10);  % 用中位数替换不会损失分布信息，比均值好多了
    data(:,i) = filloutliers(d,'nearest','movmedian',500);
    
end

acc = data(:,1:3);
gyro = data(:,4:6);
mag = data(:,7:9);
% x = acc(:,1);
% y = acc(:,2);
% z = acc(:,3);
% SVM = (acc(:,1).^2+acc(:,2).^2+acc(:,3).^2).^0.5; % signal vector magnitude

subplot(2,2,1);plot(data(:,1:3));grid on;xlabel('时间/s');ylabel('加速度/g');title('加速度曲线');
subplot(2,2,2);plot(data(:,4:6));grid on;xlabel('时间/s');ylabel('角速度/°/s');title('角速度曲线');
subplot(2,2,3);plot(data(:,7:9));grid on;xlabel('时间/s');ylabel('角度/°');title('角度曲线');

% 时域特征 计算
% y=normrnd(600,1,1000,1);   %生成平均值为600，标准差为1的1000*1的矩阵
% plot(y);
% ma = max(y); 			%最大值
% mi = min(y); 			%最小值	
% me = mean(y); 			%平均值
% pk = ma-mi;			    %峰-峰值
% av = mean(abs(y));		%绝对值的平均值(整流平均值)
% va = var(y);			%方差
% st = std(y);			%标准差
% ku = kurtosis(y);		%峭度
% sk = skewness(y);       %偏度
% rm = rms(y);			%均方根
% S = rm/av;			    %波形因子
% C = pk/rm;		    	%峰值因子
% I = pk/av;		    	%脉冲因子
% xr = mean(sqrt(abs(y)))^2;
% L = pk/xr;		    	%裕度因子
