import usb
import usb.core

busses = usb.busses()

print(busses)

for bus in busses:
    print(bus)
    devices = bus.devices
    for dev in devices:
        print('name:',dev.filename)
        print('manu:',dev.manufacturer)
