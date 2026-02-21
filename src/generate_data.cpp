#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main() {
    ofstream file("../data/traffic_data.csv");
    file << "hour,speed,density,label\n";

    srand(time(0));

    for (int i = 0; i < 20000; i++) {
        int hour = rand() % 24;                 // time of day
        double speed = 20 + rand() % 80;        // km/h
        double density = rand() % 200;          // vehicles/km

        // Simple real-life rule
        int label = (speed < 40 && density > 100) ? 1 : 0;

        file << hour << "," << speed << "," << density << "," << label << "\n";
    }

    file.close();
    cout << "✅ traffic_data.csv generated successfully\n";
    return 0;
}
