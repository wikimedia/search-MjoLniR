Vagrant.configure("2") do |config|

    config.vm.provider :lxc do |_lxc, override|
        override.vm.box = 'LEAP/jessie'
    end

    config.vm.provider :virtualbox do |vb, override|
        override.vm.box = 'debian/contrib-jessie64'
        vb.customize ['modifyvm', :id, '--memory', '2048']

        root_share_options = { id: 'vagrant-root' }
        root_share_options[:type] = :nfs
        root_share_options[:mount_options] = ['noatime', 'rsize=32767', 'wsize=3267', 'async', 'nolock']
        override.nfs.map_uid = Process.uid
        override.nfs.map_gid = Process.gid
        override.vm.synced_folder ".", "/vagrant", root_share_options

        override.vm.hostname = "MjoLniR"
        override.vm.network "private_network", type: "dhcp"
    end

    config.vm.provision "shell", path: "bootstrap-vm.sh"
end
