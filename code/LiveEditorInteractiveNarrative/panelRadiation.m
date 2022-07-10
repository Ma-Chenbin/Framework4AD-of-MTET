function [lTime, sRad, pRad] = panelRadiation(date, longitude, latitude, UTCoff, ...
    panelTilt, panelAzimuth)

% Initialize outputs
lTime = datetime.empty ;
lTime.TimeZone = date.TimeZone;
sRad = [] ;
pRad = [] ;

% Calculate day of the year for the startDate and endDate
DOY = caldays(between(datetime(date.Year,1,1), date, 'Day')) + 1;

% Calculate the solar declination for the day
delta = asind(sind(23.45)*sind(360*(DOY - 81)/365));

% Calculate the solar time correction the day
sCorr = solarCorrection(DOY, longitude, str2double(UTCoff));

% Calculate sunrise and sunet for the day
midnight = dateshift(date,'start','day');
sunrise  = 12 - acosd(-tand(latitude)*tand(delta))/15 - sCorr/60;
sunset   = 12 + acosd(-tand(latitude)*tand(delta))/15 - sCorr/60;

%     fprintf('Sunrise = %s\nSunset  = %s\n', timeofday(midnight + hours(sunrise)), ...
%         timeofday(midnight + hours(sunset)))

% Loop through the hours between sunrise and sunset
firstHour = ceil(sunrise);
lastHour = floor(sunset);
for j = 1:lastHour-firstHour+1
    
    % Calculate local and solar time
    lTime(end+1) = midnight + hours(firstHour+j-1);
    sTime = lTime(j) + minutes(sCorr);
    omega = 15*(sTime.Hour + sTime.Minute/60 - 12);
    
    % Calculate solar elevation (alpha) and azimuth (gamma)
    alpha = asind(sind(delta)*sind(latitude) + cosd(delta)*cosd(latitude)*cosd(omega));
    gamma = acosd((sind(delta)*cosd(latitude) - cosd(delta)*sind(latitude)*cosd(omega))/cosd(alpha));
    if (hour(sTime) >= 12) && (omega >= 0)
        gamma = 360 - gamma;
    end
    
    % Calculate airmass and maximum solar radiation
    AM = 1/(cosd(90-alpha) + 0.50572*(6.07955+alpha)^-1.6354);
    sRad(end+1) = 1.353*0.7^(AM^0.678);
    
    % Calculate panel radiation
    pRad(end+1) = sRad(end)*max(0,(cosd(alpha)*sind(panelTilt)*cosd(panelAzimuth-gamma) + sind(alpha)*cosd(panelTilt)));
    
end

end