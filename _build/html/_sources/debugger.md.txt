# Debugger

The debugger is the most important tool to have to expediate testing and development of code. 
It allows the coder to quickly find and resolve bugs in the code.


## Configuring pyliqtr to be debugged

Visual Code Studio (VCS) has a built in debugging tool, but it must be turned on in order to be utilized.
To do setup the debugger do the following:

1. Click the "Run and Debug" icon on the far left vertical toolbar list (it looks like a triangle with a ladybug)
2. On the nex GUI click the "launch.json" file link.
3. Paste the following into the launch.json file (this file lives at the root project directory)

{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "console": "integratedTerminal",
            "module": "pytest",
            "args": [
                "-s",
                "-m run_this_test"
            ],
            "justMyCode": false,
            "env": {
                "PYTHONPATH": "${workspaceRoot}"
            }
        }
    ]
}

4. Save the launch.json file.
5. The GUI to the left will change and at the top, next to the "Run" triangle, there will be a dropdown with "Python: Current File (pyliqtr)" visible.

If it was configured correctly you'll see the env for your main conda environment started and the Terminal session will automatically change to wherever pyliqtr lives.


## Configuring specific tests to be debugged

In the launch.json file you copied there was a parameter:
            "args": [
                "-s",
                "-m run_this_test"
            ],
the -m part of this parameter tells the debugger to only run tests that have been tagged with the tag "run_this_test".
You can tag 1 or more tests with this tag. One of the features that makes a debugger so powerful is the ability to run 1 single test.
This allows for debugging without having to wade through logs and/or errors from potentianlly dozens or hundreds of other tests.

In order to tag a test to be run this way, you must add the following annotation before it:
    @pytest.mark.run_this_test
Note: you can change the tag "run_this_test" to be whatever you want, but the tag in the test file must match the tag in the launch.json file.

an example of a marked test is below:

    @pytest.mark.run_this_test
    def test_default_settings(self, settings_obj):
        assert settings_obj.bias_list[0] == "qubit-01_Z"
        assert settings_obj.bias_list[1] == "qubit-01_X"

        assert len(settings_obj.points) == len(settings_obj.bias_list)
        assert all([settings_obj.points[n] == 100 for n in range(2)])

        assert len(settings_obj.amplitude_list) == len(settings_obj.bias_list)
        assert all([settings_obj.amplitude_list[n] == 1.5 for n in range(2)])


## Running the debugger

Once the launch.json file is updated and your tests are tagged, you are ready to debug.
Now you can run the debugger by clicking the green triangle in the Debugger pane, and you'll see the session running in the Terminal.
You can now set breakpoints wherever you need them in the various files you are testing.
To set a breakpoint simply click the space to the left of the line number in the file you'd like to break at, on the line you'd
like to stop program execution at. You can then use the "Variables" frame in the Debugger pane to inspect all the various objects
that are being utilized by code you are testing.
You can also see the execution stack, in reverse order from top to bottom, in the "Call Stack" frame.
