clear;
close all;
% newData = importdata('I:\posture_tremor\postural_action_video\0921��_��ǰһ��\200921094020.txt', '\t', 2);
newData = readtable('I:\posture_tremor\postural_action_video\1028��_����һ��\201028104723.txt');
rawdata = table2array(newData(:,4:12));
% ��ֵ��Ĵ���
% newData = importdata('H:\rest_tremor\data\1202��_ɸ��\201019152517.txt', '\t');


% ʹ��Фά�շ����������Ÿ��ʣ��޳��쳣ֵ
[m,n] = size(rawdata);
% w = 1 + 0.4*log(m);    % Фά��ϵ�������Ƽ��㹫ʽ��
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
    d = fillmissing(d,'movmedian',10);  % ����λ���滻������ʧ�ֲ���Ϣ���Ⱦ�ֵ�ö���
    data(:,i) = filloutliers(d,'nearest','movmedian',500);
    
end

acc = data(:,1:3);
gyro = data(:,4:6);
mag = data(:,7:9);
% x = acc(:,1);
% y = acc(:,2);
% z = acc(:,3);
% SVM = (acc(:,1).^2+acc(:,2).^2+acc(:,3).^2).^0.5; % signal vector magnitude

subplot(2,2,1);plot(data(:,1:3));grid on;xlabel('ʱ��/s');ylabel('���ٶ�/g');title('���ٶ�����');
subplot(2,2,2);plot(data(:,4:6));grid on;xlabel('ʱ��/s');ylabel('���ٶ�/��/s');title('���ٶ�����');
subplot(2,2,3);plot(data(:,7:9));grid on;xlabel('ʱ��/s');ylabel('�Ƕ�/��');title('�Ƕ�����');

% ʱ������ ����
% y=normrnd(600,1,1000,1);   %����ƽ��ֵΪ600����׼��Ϊ1��1000*1�ľ���
% plot(y);
% ma = max(y); 			%���ֵ
% mi = min(y); 			%��Сֵ	
% me = mean(y); 			%ƽ��ֵ
% pk = ma-mi;			    %��-��ֵ
% av = mean(abs(y));		%����ֵ��ƽ��ֵ(����ƽ��ֵ)
% va = var(y);			%����
% st = std(y);			%��׼��
% ku = kurtosis(y);		%�Ͷ�
% sk = skewness(y);       %ƫ��
% rm = rms(y);			%������
% S = rm/av;			    %��������
% C = pk/rm;		    	%��ֵ����
% I = pk/av;		    	%��������
% xr = mean(sqrt(abs(y)))^2;
% L = pk/xr;		    	%ԣ������
