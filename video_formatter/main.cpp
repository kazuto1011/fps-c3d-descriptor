#include <iostream>
#include <sstream>

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/string/predicate.hpp>

using namespace std;
namespace fs = boost::filesystem;
namespace algo = boost::algorithm;

const fs::path INPUT_PATH = "/home/kazuto/egocentric_video0/";
const fs::path OUTPUT_PATH = "/home/kazuto/egocentric_video/";

int main()
{
    cout << "Video Formatter for C3D" << endl;
    BOOST_FOREACH(const fs::path& categories_path,
                  std::make_pair(fs::directory_iterator(INPUT_PATH) , fs::directory_iterator()))
    {
        std::string category = categories_path.filename().string();
        cout << "category: " << category << endl;

        int idx = 0;

        BOOST_FOREACH(const fs::path& recursive_path,
                      std::make_pair(fs::recursive_directory_iterator(categories_path.string()) , fs::recursive_directory_iterator()))
        {
            if (recursive_path.extension() == ".avi")
            {
                std::stringstream cmd, new_name;
                fs::path output_path = OUTPUT_PATH;

                //create a new name
                new_name << category << "_" << idx << ".avi";
                cout << new_name.str() << endl;

                //create a directory at the output path
                output_path /= category;
                fs::create_directory(output_path);

                //copying
                output_path /= new_name.str();
                cmd << "cp " << recursive_path << " " << output_path;
                cout << cmd.str() << endl;
                system(cmd.str().c_str());

                idx++;
            }
        }
    }

    return 0;
}

